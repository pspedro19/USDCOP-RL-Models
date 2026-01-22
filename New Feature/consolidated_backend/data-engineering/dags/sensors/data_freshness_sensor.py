# data-engineering/dags/sensors/data_freshness_sensor.py
"""
Data Freshness Sensor for Airflow.

Custom sensor that verifies data is fresh (less than N days old)
before allowing downstream tasks to proceed.

Features:
- Configurable freshness threshold
- Multiple data source support (PostgreSQL, CSV, Parquet)
- Reschedule mode for efficient resource usage
- Detailed logging for debugging
"""

from datetime import datetime, timedelta
from typing import Optional, Callable, Any, Dict
import logging

from airflow.sensors.base import BaseSensorOperator
from airflow.utils.decorators import apply_defaults
from airflow.exceptions import AirflowSensorTimeout, AirflowSkipException

import pandas as pd


logger = logging.getLogger(__name__)


class DataFreshnessSensor(BaseSensorOperator):
    """
    Sensor that waits until data is fresh (within specified days).

    This sensor checks if the most recent data in a source is within
    the specified freshness threshold. It's designed for use cases where
    you need to ensure data pipelines only proceed with recent data.

    Parameters:
        data_source: Path to CSV/Parquet file or PostgreSQL connection string
        date_column: Name of the date column to check
        max_days: Maximum acceptable age in days (default: 7)
        table_name: Table name if using PostgreSQL
        query: Custom SQL query (optional, overrides table_name)
        poke_interval: Time between checks in seconds (default: 300 = 5 minutes)
        timeout: Maximum time to wait in seconds (default: 3600 = 1 hour)
        mode: Sensor mode ('poke' or 'reschedule', default: 'reschedule')
        soft_fail: If True, skip task instead of failing on timeout

    Example:
        >>> freshness_sensor = DataFreshnessSensor(
        ...     task_id='check_data_freshness',
        ...     data_source='/path/to/data.csv',
        ...     date_column='Date',
        ...     max_days=7,
        ...     poke_interval=300,
        ...     timeout=3600,
        ...     mode='reschedule'
        ... )

    Example with PostgreSQL:
        >>> freshness_sensor = DataFreshnessSensor(
        ...     task_id='check_db_freshness',
        ...     data_source='postgresql://user:pass@host:5432/db',
        ...     table_name='core.features_ml',
        ...     date_column='date',
        ...     max_days=3
        ... )
    """

    template_fields = ('data_source', 'table_name', 'query', 'date_column')
    ui_color = '#66c2ff'
    ui_fgcolor = '#000000'

    @apply_defaults
    def __init__(
        self,
        data_source: str,
        date_column: str,
        max_days: int = 7,
        table_name: Optional[str] = None,
        query: Optional[str] = None,
        poke_interval: int = 300,  # 5 minutes
        timeout: int = 3600,  # 1 hour
        mode: str = 'reschedule',
        soft_fail: bool = False,
        on_failure_callback: Optional[Callable] = None,
        reference_date: Optional[datetime] = None,
        *args,
        **kwargs
    ):
        super().__init__(
            poke_interval=poke_interval,
            timeout=timeout,
            mode=mode,
            soft_fail=soft_fail,
            *args,
            **kwargs
        )
        self.data_source = data_source
        self.date_column = date_column
        self.max_days = max_days
        self.table_name = table_name
        self.query = query
        self.on_failure_callback = on_failure_callback
        self.reference_date = reference_date
        self._last_check_result = None

    def poke(self, context: Dict[str, Any]) -> bool:
        """
        Check if data is fresh.

        Returns:
            True if data is fresh (within max_days), False otherwise
        """
        logger.info(f"Checking data freshness for source: {self.data_source}")
        logger.info(f"Max allowed age: {self.max_days} days")

        try:
            # Get latest date from data source
            latest_date = self._get_latest_date()

            if latest_date is None:
                logger.warning("Could not determine latest date from data source")
                self._last_check_result = {
                    'success': False,
                    'error': 'No date found'
                }
                return False

            # Calculate age
            reference = self.reference_date or datetime.now()

            # Handle timezone-aware datetimes
            if latest_date.tzinfo is not None:
                if reference.tzinfo is None:
                    reference = reference.replace(tzinfo=latest_date.tzinfo)
            else:
                if hasattr(reference, 'tzinfo') and reference.tzinfo is not None:
                    reference = reference.replace(tzinfo=None)

            age_delta = reference - latest_date
            age_days = age_delta.total_seconds() / (24 * 3600)

            logger.info(f"Latest data date: {latest_date}")
            logger.info(f"Data age: {age_days:.2f} days")

            is_fresh = age_days <= self.max_days

            self._last_check_result = {
                'success': True,
                'latest_date': str(latest_date),
                'age_days': age_days,
                'is_fresh': is_fresh,
                'max_days': self.max_days
            }

            if is_fresh:
                logger.info(f"Data is FRESH (age: {age_days:.2f} days <= {self.max_days} days)")
            else:
                logger.warning(
                    f"Data is STALE (age: {age_days:.2f} days > {self.max_days} days)"
                )

            # Push results to XCom for downstream tasks
            if context and 'ti' in context:
                context['ti'].xcom_push(
                    key='freshness_check_result',
                    value=self._last_check_result
                )

            return is_fresh

        except Exception as e:
            logger.error(f"Error checking data freshness: {e}")
            self._last_check_result = {
                'success': False,
                'error': str(e)
            }

            if self.on_failure_callback:
                self.on_failure_callback(context, e)

            return False

    def _get_latest_date(self) -> Optional[datetime]:
        """
        Get the latest date from the data source.

        Returns:
            Latest datetime or None if unable to determine
        """
        # Check if PostgreSQL connection
        if self.data_source.startswith('postgresql://'):
            return self._get_latest_from_postgres()

        # Check file type
        if self.data_source.endswith('.csv'):
            return self._get_latest_from_csv()
        elif self.data_source.endswith('.parquet'):
            return self._get_latest_from_parquet()
        else:
            # Try to auto-detect
            try:
                return self._get_latest_from_csv()
            except Exception:
                try:
                    return self._get_latest_from_parquet()
                except Exception:
                    return self._get_latest_from_postgres()

    def _get_latest_from_csv(self) -> Optional[datetime]:
        """Get latest date from CSV file."""
        try:
            # Read only the date column for efficiency
            df = pd.read_csv(
                self.data_source,
                usecols=[self.date_column],
                parse_dates=[self.date_column]
            )

            if df.empty:
                return None

            latest = df[self.date_column].max()
            return latest.to_pydatetime() if pd.notna(latest) else None

        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            raise

    def _get_latest_from_parquet(self) -> Optional[datetime]:
        """Get latest date from Parquet file."""
        try:
            df = pd.read_parquet(
                self.data_source,
                columns=[self.date_column]
            )

            if df.empty:
                return None

            latest = df[self.date_column].max()
            return latest.to_pydatetime() if pd.notna(latest) else None

        except Exception as e:
            logger.error(f"Error reading Parquet: {e}")
            raise

    def _get_latest_from_postgres(self) -> Optional[datetime]:
        """Get latest date from PostgreSQL."""
        try:
            from sqlalchemy import create_engine

            # Build query
            if self.query:
                sql = self.query
            elif self.table_name:
                sql = f"SELECT MAX({self.date_column}) as latest_date FROM {self.table_name}"
            else:
                raise ValueError("Either query or table_name must be provided for PostgreSQL")

            # Execute query
            engine = create_engine(self.data_source)
            with engine.connect() as conn:
                result = pd.read_sql(sql, conn)

            if result.empty:
                return None

            latest = result.iloc[0, 0]

            # Convert to datetime if needed
            if isinstance(latest, str):
                latest = pd.to_datetime(latest)
            elif isinstance(latest, pd.Timestamp):
                latest = latest.to_pydatetime()

            return latest

        except Exception as e:
            logger.error(f"Error querying PostgreSQL: {e}")
            raise

    def execute(self, context: Dict[str, Any]) -> None:
        """
        Execute the sensor.

        Overrides base execute to add custom logging and callbacks.
        """
        try:
            super().execute(context)

            logger.info("Data freshness check PASSED")

            # Log final result
            if self._last_check_result:
                logger.info(f"Final check result: {self._last_check_result}")

        except AirflowSensorTimeout as e:
            logger.error(f"Data freshness sensor TIMED OUT after {self.timeout} seconds")

            if self.soft_fail:
                raise AirflowSkipException(
                    f"Data not fresh after {self.timeout}s - skipping downstream tasks"
                )

            if self.on_failure_callback:
                self.on_failure_callback(context, e)

            raise

    def get_last_check_result(self) -> Optional[Dict[str, Any]]:
        """Get the result of the last freshness check."""
        return self._last_check_result


class MultiSourceFreshnessSensor(BaseSensorOperator):
    """
    Sensor that checks freshness across multiple data sources.

    All sources must be fresh for the sensor to return True.

    Parameters:
        sources: List of dictionaries with source configurations
        poke_interval: Time between checks in seconds
        timeout: Maximum time to wait in seconds
        mode: Sensor mode ('poke' or 'reschedule')
        require_all: If True, all sources must be fresh. If False, any source.

    Example:
        >>> sensor = MultiSourceFreshnessSensor(
        ...     task_id='check_all_sources',
        ...     sources=[
        ...         {'path': '/data/prices.csv', 'date_col': 'Date', 'max_days': 3},
        ...         {'path': '/data/features.csv', 'date_col': 'Date', 'max_days': 7},
        ...     ],
        ...     require_all=True
        ... )
    """

    template_fields = ('sources',)
    ui_color = '#66c2ff'

    @apply_defaults
    def __init__(
        self,
        sources: list,
        poke_interval: int = 300,
        timeout: int = 3600,
        mode: str = 'reschedule',
        require_all: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(
            poke_interval=poke_interval,
            timeout=timeout,
            mode=mode,
            *args,
            **kwargs
        )
        self.sources = sources
        self.require_all = require_all
        self._results = {}

    def poke(self, context: Dict[str, Any]) -> bool:
        """Check freshness of all sources."""
        logger.info(f"Checking freshness of {len(self.sources)} sources")

        fresh_count = 0
        total_sources = len(self.sources)

        for i, source_config in enumerate(self.sources):
            path = source_config.get('path', source_config.get('data_source'))
            date_col = source_config.get('date_col', source_config.get('date_column', 'Date'))
            max_days = source_config.get('max_days', 7)

            logger.info(f"Checking source {i+1}/{total_sources}: {path}")

            try:
                # Use existing sensor logic
                sensor = DataFreshnessSensor(
                    task_id=f'temp_sensor_{i}',
                    data_source=path,
                    date_column=date_col,
                    max_days=max_days
                )

                is_fresh = sensor.poke(context)
                self._results[path] = sensor.get_last_check_result()

                if is_fresh:
                    fresh_count += 1
                    logger.info(f"Source {path} is FRESH")
                else:
                    logger.warning(f"Source {path} is STALE")

            except Exception as e:
                logger.error(f"Error checking source {path}: {e}")
                self._results[path] = {'success': False, 'error': str(e)}

        # Push results to XCom
        if context and 'ti' in context:
            context['ti'].xcom_push(key='multi_source_results', value=self._results)

        # Determine overall result
        if self.require_all:
            return fresh_count == total_sources
        else:
            return fresh_count > 0
