"""
Custom Airflow Sensors for Event-Driven Data Processing
=========================================================

Replaces fixed 5-minute schedules with data-driven execution.
Sensors poke the database to check if new data is available before
allowing downstream tasks to execute.

Pattern:
    L0 (fixed schedule) -> inserts OHLCV
    L1 (sensor waits for OHLCV) -> calculates features
    L5 (sensor waits for features) -> runs inference

Benefits:
1. No schedule drift - runs exactly when data is ready
2. No overlapping jobs - sensor blocks until data arrives
3. Event-driven without extra infrastructure (uses PostgreSQL)
4. Automatic backpressure if upstream is slow

Author: Pedro @ Lean Tech Solutions
Version: 1.0.0
Created: 2025-01-12
"""

from datetime import datetime, timedelta
from typing import Optional
import logging

from airflow.sensors.base import BaseSensorOperator
from airflow.utils.decorators import apply_defaults
import psycopg2

from utils.dag_common import get_db_connection

logger = logging.getLogger(__name__)


class NewOHLCVBarSensor(BaseSensorOperator):
    """
    Sensor that waits for new OHLCV bar data in usdcop_m5_ohlcv table.

    Used by L1 (Feature Calculation) to wait for L0 (OHLCV Acquisition)
    to complete before processing.

    Poke Logic:
        1. Get latest bar timestamp from usdcop_m5_ohlcv
        2. Compare with last known timestamp (from XCom or calculated)
        3. Return True if new bar detected, False otherwise

    Args:
        table_name: Table to monitor (default: usdcop_m5_ohlcv)
        symbol: Symbol to filter (default: USD/COP)
        max_staleness_minutes: Max age of data to consider "fresh" (default: 10)
        poke_interval: Seconds between checks (default: 30)
        timeout: Max seconds to wait (default: 300 = 5 minutes)
        mode: 'poke' or 'reschedule' (default: 'poke')

    Example:
        sensor_ohlcv = NewOHLCVBarSensor(
            task_id='wait_for_ohlcv',
            poke_interval=30,
            timeout=300,
        )
        sensor_ohlcv >> task_calculate_features
    """

    template_fields = ('table_name', 'symbol')

    @apply_defaults
    def __init__(
        self,
        table_name: str = 'usdcop_m5_ohlcv',
        symbol: str = 'USD/COP',
        max_staleness_minutes: int = 10,
        poke_interval: int = 30,
        timeout: int = 300,
        mode: str = 'poke',
        **kwargs
    ):
        super().__init__(
            poke_interval=poke_interval,
            timeout=timeout,
            mode=mode,
            **kwargs
        )
        self.table_name = table_name
        self.symbol = symbol
        self.max_staleness_minutes = max_staleness_minutes
        self._last_seen_time: Optional[datetime] = None

    def poke(self, context) -> bool:
        """
        Check if new OHLCV bar is available.

        Returns True if:
        1. New bar exists since last check, OR
        2. Latest bar is within staleness window and hasn't been processed
        """
        conn = None
        try:
            conn = get_db_connection()
            cur = conn.cursor()

            # Get latest bar timestamp
            cur.execute(f"""
                SELECT MAX(time) as latest_time
                FROM {self.table_name}
                WHERE symbol = %s
            """, (self.symbol,))

            result = cur.fetchone()
            latest_time = result[0] if result else None

            cur.close()
            conn.close()

            if latest_time is None:
                logger.warning(f"No data found in {self.table_name}")
                return False

            # Get last processed time from XCom (if available)
            ti = context.get('ti')
            last_processed = None
            if ti:
                last_processed = ti.xcom_pull(
                    key='last_processed_ohlcv_time',
                    include_prior_dates=True
                )

            # If no prior processing record, use internal tracker
            if last_processed is None:
                last_processed = self._last_seen_time

            # Check staleness
            now = datetime.utcnow()
            staleness = (now - latest_time.replace(tzinfo=None)).total_seconds() / 60

            if staleness > self.max_staleness_minutes:
                logger.info(
                    f"Latest OHLCV bar is {staleness:.1f} minutes old "
                    f"(max: {self.max_staleness_minutes}), waiting..."
                )
                return False

            # Check if new data since last processing
            if last_processed is not None:
                if isinstance(last_processed, str):
                    last_processed = datetime.fromisoformat(last_processed)

                if latest_time <= last_processed:
                    logger.info(
                        f"No new OHLCV data. Latest: {latest_time}, "
                        f"Last processed: {last_processed}"
                    )
                    return False

            # New data detected
            logger.info(f"New OHLCV bar detected at {latest_time}")

            # Store for next check
            self._last_seen_time = latest_time

            # Push to XCom for downstream tasks
            if ti:
                ti.xcom_push(key='detected_ohlcv_time', value=str(latest_time))

            return True

        except Exception as e:
            logger.error(f"Error in NewOHLCVBarSensor: {e}")
            if conn:
                conn.close()
            return False


class NewFeatureBarSensor(BaseSensorOperator):
    """
    Sensor that waits for new feature data in inference_features_5m table.

    Used by L5 (Inference) to wait for L1 (Feature Calculation) to complete
    before running model inference.

    Poke Logic:
        1. Get latest feature timestamp from inference_features_5m
        2. Optionally verify feature completeness (no NULLs in critical columns)
        3. Compare with last known timestamp
        4. Return True if new complete features detected

    Args:
        table_name: Table to monitor (default: inference_features_5m)
        require_complete: Require all features to be non-NULL (default: True)
        critical_features: List of features that must be non-NULL
        max_staleness_minutes: Max age of data to consider "fresh" (default: 10)
        poke_interval: Seconds between checks (default: 30)
        timeout: Max seconds to wait (default: 300 = 5 minutes)
        mode: 'poke' or 'reschedule' (default: 'poke')

    Example:
        sensor_features = NewFeatureBarSensor(
            task_id='wait_for_features',
            require_complete=True,
            poke_interval=30,
            timeout=300,
        )
        sensor_features >> task_run_inference
    """

    template_fields = ('table_name',)

    # Critical features that must be present for valid inference
    DEFAULT_CRITICAL_FEATURES = [
        'log_ret_5m',
        'log_ret_1h',
        'rsi_9',
        'dxy_z',
        'vix_z',
        'rate_spread',
    ]

    @apply_defaults
    def __init__(
        self,
        table_name: str = 'inference_features_5m',
        require_complete: bool = True,
        critical_features: Optional[list] = None,
        max_staleness_minutes: int = 10,
        poke_interval: int = 30,
        timeout: int = 300,
        mode: str = 'poke',
        **kwargs
    ):
        super().__init__(
            poke_interval=poke_interval,
            timeout=timeout,
            mode=mode,
            **kwargs
        )
        self.table_name = table_name
        self.require_complete = require_complete
        self.critical_features = critical_features or self.DEFAULT_CRITICAL_FEATURES
        self.max_staleness_minutes = max_staleness_minutes
        self._last_seen_time: Optional[datetime] = None

    def poke(self, context) -> bool:
        """
        Check if new feature data is available and complete.

        Returns True if:
        1. New feature row exists since last check, AND
        2. Critical features are non-NULL (if require_complete=True)
        """
        conn = None
        try:
            conn = get_db_connection()
            cur = conn.cursor()

            # Build completeness check SQL
            if self.require_complete:
                completeness_checks = " AND ".join(
                    [f"{feat} IS NOT NULL" for feat in self.critical_features]
                )
                where_clause = f"WHERE {completeness_checks}"
            else:
                where_clause = ""

            # Get latest complete feature row
            cur.execute(f"""
                SELECT MAX(time) as latest_time
                FROM {self.table_name}
                {where_clause}
            """)

            result = cur.fetchone()
            latest_time = result[0] if result else None

            cur.close()
            conn.close()

            if latest_time is None:
                logger.warning(
                    f"No {'complete ' if self.require_complete else ''}"
                    f"features found in {self.table_name}"
                )
                return False

            # Get last processed time from XCom (if available)
            ti = context.get('ti')
            last_processed = None
            if ti:
                last_processed = ti.xcom_pull(
                    key='last_processed_feature_time',
                    include_prior_dates=True
                )

            # If no prior processing record, use internal tracker
            if last_processed is None:
                last_processed = self._last_seen_time

            # Check staleness
            now = datetime.utcnow()
            staleness = (now - latest_time.replace(tzinfo=None)).total_seconds() / 60

            if staleness > self.max_staleness_minutes:
                logger.info(
                    f"Latest features are {staleness:.1f} minutes old "
                    f"(max: {self.max_staleness_minutes}), waiting..."
                )
                return False

            # Check if new data since last processing
            if last_processed is not None:
                if isinstance(last_processed, str):
                    last_processed = datetime.fromisoformat(last_processed)

                if latest_time <= last_processed:
                    logger.info(
                        f"No new feature data. Latest: {latest_time}, "
                        f"Last processed: {last_processed}"
                    )
                    return False

            # New complete features detected
            logger.info(f"New complete features detected at {latest_time}")

            # Store for next check
            self._last_seen_time = latest_time

            # Push to XCom for downstream tasks
            if ti:
                ti.xcom_push(key='detected_feature_time', value=str(latest_time))

            return True

        except Exception as e:
            logger.error(f"Error in NewFeatureBarSensor: {e}")
            if conn:
                conn.close()
            return False


class DataFreshnessGuard:
    """
    Utility class for checking data freshness without blocking.

    Can be used in tasks that need to verify data freshness without
    using a full sensor (e.g., for logging or conditional logic).

    Example:
        guard = DataFreshnessGuard()
        if not guard.is_ohlcv_fresh():
            logging.warning("OHLCV data is stale!")
            return {'status': 'skipped', 'reason': 'stale_data'}
    """

    def __init__(self, max_staleness_minutes: int = 10):
        self.max_staleness_minutes = max_staleness_minutes

    def is_ohlcv_fresh(self, symbol: str = 'USD/COP') -> bool:
        """Check if OHLCV data is fresh."""
        conn = None
        try:
            conn = get_db_connection()
            cur = conn.cursor()

            cur.execute("""
                SELECT MAX(time) as latest_time
                FROM usdcop_m5_ohlcv
                WHERE symbol = %s
            """, (symbol,))

            result = cur.fetchone()
            latest_time = result[0] if result else None

            cur.close()
            conn.close()

            if latest_time is None:
                return False

            now = datetime.utcnow()
            staleness = (now - latest_time.replace(tzinfo=None)).total_seconds() / 60

            return staleness <= self.max_staleness_minutes

        except Exception as e:
            logger.error(f"Error checking OHLCV freshness: {e}")
            if conn:
                conn.close()
            return False

    def is_features_fresh(self) -> bool:
        """Check if feature data is fresh."""
        conn = None
        try:
            conn = get_db_connection()
            cur = conn.cursor()

            cur.execute("""
                SELECT MAX(time) as latest_time
                FROM inference_features_5m
            """)

            result = cur.fetchone()
            latest_time = result[0] if result else None

            cur.close()
            conn.close()

            if latest_time is None:
                return False

            now = datetime.utcnow()
            staleness = (now - latest_time.replace(tzinfo=None)).total_seconds() / 60

            return staleness <= self.max_staleness_minutes

        except Exception as e:
            logger.error(f"Error checking features freshness: {e}")
            if conn:
                conn.close()
            return False

    def get_data_lag(self) -> dict:
        """Get lag information for all data sources."""
        conn = None
        try:
            conn = get_db_connection()
            cur = conn.cursor()

            # Get OHLCV lag
            cur.execute("""
                SELECT MAX(time) FROM usdcop_m5_ohlcv WHERE symbol = 'USD/COP'
            """)
            ohlcv_time = cur.fetchone()[0]

            # Get features lag
            cur.execute("""
                SELECT MAX(time) FROM inference_features_5m
            """)
            features_time = cur.fetchone()[0]

            # Get signals lag
            cur.execute("""
                SELECT MAX(created_at) FROM trading_signals
            """)
            signals_time = cur.fetchone()[0]

            cur.close()
            conn.close()

            now = datetime.utcnow()

            def calc_lag(t):
                if t is None:
                    return None
                return (now - t.replace(tzinfo=None)).total_seconds() / 60

            return {
                'ohlcv_lag_minutes': calc_lag(ohlcv_time),
                'features_lag_minutes': calc_lag(features_time),
                'signals_lag_minutes': calc_lag(signals_time),
                'ohlcv_latest': str(ohlcv_time) if ohlcv_time else None,
                'features_latest': str(features_time) if features_time else None,
                'signals_latest': str(signals_time) if signals_time else None,
            }

        except Exception as e:
            logger.error(f"Error getting data lag: {e}")
            if conn:
                conn.close()
            return {}
