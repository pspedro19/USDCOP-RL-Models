"""
L1 Features Sensor - Wait for L1 Feature Pipeline Output
=========================================================

Custom Airflow sensor that waits for features from the L1 pipeline
to be available before allowing downstream tasks (L5 inference) to run.

This sensor provides:
- Feature availability checking by timestamp
- Norm stats hash validation (optional)
- Freshness checking with configurable max age
- Integration with FeatureReader for database queries

Usage in DAG:
    from sensors.feature_sensor import L1FeaturesSensor

    sensor = L1FeaturesSensor(
        task_id='wait_for_l1_features',
        symbol='USD/COP',
        max_age_minutes=10,
        expected_norm_stats_hash='abc123...',  # Optional
        poke_interval=30,
        timeout=300,
    )

    sensor >> inference_task

Author: Trading Team
Version: 1.0.0
Created: 2026-01-17
"""

from datetime import datetime, timedelta
from typing import Optional
import logging
import sys
from pathlib import Path

from airflow.sensors.base import BaseSensorOperator
from airflow.utils.decorators import apply_defaults

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import FeatureReader
try:
    from src.feature_store.feature_reader import FeatureReader
    FEATURE_READER_AVAILABLE = True
except ImportError as e:
    FEATURE_READER_AVAILABLE = False
    logging.warning(f"FeatureReader not available: {e}")

logger = logging.getLogger(__name__)


class L1FeaturesSensor(BaseSensorOperator):
    """
    Sensor that waits for L1 feature pipeline output to be available.

    This sensor integrates with the FeatureReader to check if features
    have been computed by the L1 pipeline for the current execution date.
    It supports:

    1. Feature availability checking
    2. Norm stats hash validation (for training/inference parity)
    3. Configurable freshness requirements
    4. XCom integration for passing feature metadata to downstream tasks

    Args:
        symbol: Trading symbol to check features for (default: "USD/COP")
        expected_norm_stats_hash: Optional hash to validate normalization stats
        max_age_minutes: Maximum age of features to consider valid (default: 10)
        poke_interval: Seconds between checks (default: 30)
        timeout: Maximum seconds to wait (default: 300)
        mode: Sensor mode - 'poke' or 'reschedule' (default: 'poke')

    XCom outputs:
        - feature_timestamp: Timestamp of detected features
        - feature_age_seconds: Age of features when detected
        - builder_version: Version of builder used to create features
        - norm_stats_hash_match: Whether hash matched expected (if provided)

    Example:
        # Basic usage
        sensor = L1FeaturesSensor(
            task_id='wait_for_features',
            symbol='USD/COP',
        )

        # With hash validation
        sensor = L1FeaturesSensor(
            task_id='wait_for_features',
            symbol='USD/COP',
            expected_norm_stats_hash='a1b2c3d4...',
            max_age_minutes=5,
        )
    """

    template_fields = ('symbol', 'expected_norm_stats_hash')

    @apply_defaults
    def __init__(
        self,
        symbol: str = 'USD/COP',
        expected_norm_stats_hash: Optional[str] = None,
        max_age_minutes: int = 10,
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
        self.symbol = symbol
        self.expected_norm_stats_hash = expected_norm_stats_hash
        self.max_age_minutes = max_age_minutes

        # Initialize feature reader
        if FEATURE_READER_AVAILABLE:
            self._reader = FeatureReader()
        else:
            self._reader = None

    def poke(self, context) -> bool:
        """
        Check if L1 features are available for the execution date.

        This method:
        1. Gets the execution date from context
        2. Queries for features using FeatureReader
        3. Validates freshness against max_age_minutes
        4. Optionally validates norm_stats_hash
        5. Pushes feature metadata to XCom

        Args:
            context: Airflow task context

        Returns:
            bool: True if features are available and valid, False otherwise
        """
        if self._reader is None:
            logger.error("FeatureReader not available - cannot check features")
            return False

        # Get execution date from context
        execution_date = context.get('execution_date') or datetime.utcnow()
        if hasattr(execution_date, 'datetime'):
            execution_date = execution_date.datetime()

        logger.info(
            f"Checking L1 features for {self.symbol} at {execution_date}, "
            f"max_age={self.max_age_minutes}min"
        )

        try:
            # Get latest features
            record = self._reader.get_latest_features(
                self.symbol,
                max_age_minutes=self.max_age_minutes,
            )

            if record is None:
                logger.info(f"No features found for {self.symbol}")
                return False

            # Check freshness
            if not record.is_fresh(self.max_age_minutes):
                logger.info(
                    f"Features are stale: age={record.age_seconds:.1f}s, "
                    f"max={self.max_age_minutes * 60}s"
                )
                return False

            # Validate norm_stats_hash if provided
            hash_match = None
            if self.expected_norm_stats_hash:
                matches, actual_hash = self._reader.check_norm_stats_hash(
                    self.expected_norm_stats_hash,
                    record.timestamp,
                )
                hash_match = matches

                if not matches:
                    logger.warning(
                        f"NORM STATS HASH MISMATCH: "
                        f"expected={self.expected_norm_stats_hash[:12]}..., "
                        f"actual={actual_hash[:12] if actual_hash else 'None'}. "
                        f"This may cause feature drift!"
                    )
                    # Warning only - don't block on hash mismatch
                else:
                    logger.info("Norm stats hash validated successfully")

            # Features are available!
            logger.info(
                f"L1 features detected at {record.timestamp}, "
                f"age={record.age_seconds:.1f}s, "
                f"builder={record.builder_version or 'unknown'}"
            )

            # Push to XCom for downstream tasks
            ti = context.get('ti')
            if ti:
                ti.xcom_push(key='feature_timestamp', value=str(record.timestamp))
                ti.xcom_push(key='feature_age_seconds', value=record.age_seconds)
                ti.xcom_push(key='builder_version', value=record.builder_version)
                if hash_match is not None:
                    ti.xcom_push(key='norm_stats_hash_match', value=hash_match)

            return True

        except Exception as e:
            logger.error(f"Error in L1FeaturesSensor: {e}")
            return False

    def get_feature_reader(self) -> Optional[FeatureReader]:
        """Get the underlying FeatureReader instance."""
        return self._reader


class L1FeaturesAvailableSensor(L1FeaturesSensor):
    """
    Alias for L1FeaturesSensor with a more descriptive name.

    This sensor waits for L1 features to be available before
    allowing downstream inference tasks to proceed.
    """
    pass
