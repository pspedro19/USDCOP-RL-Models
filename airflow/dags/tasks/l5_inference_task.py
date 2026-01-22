"""
L5 Inference Task - Feature Retrieval and Inference Execution
==============================================================

This module provides the L5InferenceTask class which handles:
1. Reading pre-computed features from L1 pipeline
2. Fallback feature computation using CanonicalFeatureBuilder
3. Health monitoring and alerting for L1 pipeline issues
4. Kill switch integration for emergency trading halts

The L5InferenceTask follows a "prefer L1, fallback to compute" pattern:
- First attempts to read features from the L1 feature store
- If unavailable, computes features using CanonicalFeatureBuilder
- Tracks consecutive fallbacks and sends alerts after threshold

Usage in DAG:
    from tasks.l5_inference_task import L5InferenceTask, run_l5_inference

    # Create task instance
    inference_task = L5InferenceTask()

    # Use in PythonOperator
    task = PythonOperator(
        task_id='run_inference',
        python_callable=run_l5_inference,
        provide_context=True,
    )

Author: Trading Team
Version: 1.0.0
Created: 2026-01-17
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import logging
import sys
from pathlib import Path
import numpy as np

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# =============================================================================
# IMPORT DEPENDENCIES
# =============================================================================

# Feature Reader for L1 features
try:
    from src.feature_store.feature_reader import FeatureReader, get_feature_reader
    FEATURE_READER_AVAILABLE = True
except ImportError as e:
    FEATURE_READER_AVAILABLE = False
    logger.warning(f"FeatureReader not available: {e}")
    FeatureReader = None
    get_feature_reader = None

# CanonicalFeatureBuilder for fallback computation
try:
    from src.feature_store.builders import CanonicalFeatureBuilder, BuilderContext
    CANONICAL_BUILDER_AVAILABLE = True
except ImportError as e:
    CANONICAL_BUILDER_AVAILABLE = False
    logger.warning(f"CanonicalFeatureBuilder not available: {e}")
    CanonicalFeatureBuilder = None
    BuilderContext = None

# TradingFlags for kill switch
try:
    from src.config.trading_flags import TradingFlags, get_trading_flags
    TRADING_FLAGS_AVAILABLE = True
except ImportError as e:
    TRADING_FLAGS_AVAILABLE = False
    logger.warning(f"TradingFlags not available: {e}")
    TradingFlags = None
    get_trading_flags = None

# Database utilities
try:
    from utils.dag_common import get_db_connection
    DB_UTILS_AVAILABLE = True
except ImportError:
    DB_UTILS_AVAILABLE = False
    logger.warning("dag_common utilities not available")


# =============================================================================
# CONSTANTS
# =============================================================================

# Alert threshold for consecutive L1 fallbacks
L1_FALLBACK_ALERT_THRESHOLD = 5

# Default symbol
DEFAULT_SYMBOL = "USD/COP"

# Maximum feature age in minutes
MAX_FEATURE_AGE_MINUTES = 10


# =============================================================================
# L5 INFERENCE TASK CLASS
# =============================================================================

class L5InferenceTask:
    """
    L5 Inference Task - Handles feature retrieval and fallback computation.

    This class manages the feature retrieval strategy for L5 inference:
    1. Primary: Read features from L1 feature store (FeatureReader)
    2. Fallback: Compute features using CanonicalFeatureBuilder

    It tracks L1 health and sends alerts when too many consecutive
    fallbacks occur, indicating potential L1 pipeline issues.

    Attributes:
        feature_reader: FeatureReader instance for L1 feature access
        feature_builder: CanonicalFeatureBuilder for fallback computation
        consecutive_fallbacks: Count of consecutive L1 fallbacks
        alert_threshold: Number of fallbacks before alerting
        symbol: Trading symbol (default: USD/COP)

    Example:
        task = L5InferenceTask()

        # Get features for inference
        features = task.get_features_for_inference("USD/COP", datetime.now())

        if features is not None:
            # Run inference with features
            action = model.predict(features)
    """

    def __init__(
        self,
        feature_reader: Optional['FeatureReader'] = None,
        feature_builder: Optional['CanonicalFeatureBuilder'] = None,
        alert_threshold: int = L1_FALLBACK_ALERT_THRESHOLD,
        symbol: str = DEFAULT_SYMBOL,
    ):
        """
        Initialize L5InferenceTask.

        Args:
            feature_reader: Optional FeatureReader instance. Creates default if None.
            feature_builder: Optional CanonicalFeatureBuilder for fallback. Creates default if None.
            alert_threshold: Number of consecutive fallbacks before alerting.
            symbol: Trading symbol for feature retrieval.
        """
        self.symbol = symbol
        self.alert_threshold = alert_threshold
        self.consecutive_fallbacks = 0
        self._alert_sent = False

        # Initialize FeatureReader
        if feature_reader is not None:
            self.feature_reader = feature_reader
        elif FEATURE_READER_AVAILABLE:
            self.feature_reader = get_feature_reader()
        else:
            self.feature_reader = None
            logger.warning("FeatureReader not available - fallback only mode")

        # Initialize CanonicalFeatureBuilder for fallback
        if feature_builder is not None:
            self.feature_builder = feature_builder
        elif CANONICAL_BUILDER_AVAILABLE:
            try:
                self.feature_builder = CanonicalFeatureBuilder.for_inference()
                logger.info(
                    f"Initialized CanonicalFeatureBuilder for fallback "
                    f"(hash: {self.feature_builder.get_norm_stats_hash()[:12]}...)"
                )
            except Exception as e:
                logger.error(f"Failed to initialize CanonicalFeatureBuilder: {e}")
                self.feature_builder = None
        else:
            self.feature_builder = None
            logger.warning("CanonicalFeatureBuilder not available - L1 required")

    def get_features_for_inference(
        self,
        symbol: str,
        timestamp: datetime,
        position: float = 0.0,
        bar_idx: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """
        Get features for inference, with fallback to computation.

        This method implements the feature retrieval strategy:
        1. Try to read from L1 feature store
        2. If not available, compute using CanonicalFeatureBuilder
        3. Track fallback count and send alerts if threshold exceeded

        Args:
            symbol: Trading symbol (e.g., "USD/COP")
            timestamp: Timestamp for feature retrieval
            position: Current position for state features (default: 0.0)
            bar_idx: Optional bar index for computation (default: auto-calculated)

        Returns:
            np.ndarray of shape (15,) with features, or None if failed

        Raises:
            RuntimeError: If both L1 and fallback are unavailable
        """
        # Try L1 feature store first
        features = self._try_read_from_l1(symbol, timestamp)

        if features is not None:
            # Success! Reset fallback counter
            if self.consecutive_fallbacks > 0:
                logger.info(
                    f"L1 features restored after {self.consecutive_fallbacks} fallbacks"
                )
            self.consecutive_fallbacks = 0
            self._alert_sent = False
            return features

        # L1 not available - try fallback
        logger.warning(f"L1 features not available for {symbol} at {timestamp}")
        self.consecutive_fallbacks += 1

        # Check if we need to send alert
        if self.consecutive_fallbacks >= self.alert_threshold and not self._alert_sent:
            self._send_l1_health_alert()
            self._alert_sent = True

        # Try fallback computation
        features = self._calculate_fallback(symbol, timestamp, position, bar_idx)

        if features is not None:
            logger.info(
                f"Using fallback features (consecutive fallbacks: {self.consecutive_fallbacks})"
            )
            return features

        # Both L1 and fallback failed
        logger.error("Both L1 and fallback feature retrieval failed")
        return None

    def _try_read_from_l1(
        self,
        symbol: str,
        timestamp: datetime,
    ) -> Optional[np.ndarray]:
        """
        Try to read features from L1 feature store.

        Args:
            symbol: Trading symbol
            timestamp: Timestamp to query

        Returns:
            Feature array if available, None otherwise
        """
        if self.feature_reader is None:
            return None

        try:
            record = self.feature_reader.get_features(symbol, timestamp)

            if record is None:
                return None

            # Check freshness
            if not record.is_fresh(MAX_FEATURE_AGE_MINUTES):
                logger.info(
                    f"L1 features too old: age={record.age_seconds:.1f}s, "
                    f"max={MAX_FEATURE_AGE_MINUTES * 60}s"
                )
                return None

            logger.debug(
                f"Read L1 features: timestamp={record.timestamp}, "
                f"age={record.age_seconds:.1f}s"
            )

            return record.features

        except Exception as e:
            logger.error(f"Error reading L1 features: {e}")
            return None

    def _calculate_fallback(
        self,
        symbol: str,
        timestamp: datetime,
        position: float = 0.0,
        bar_idx: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """
        Calculate features using CanonicalFeatureBuilder as fallback.

        This method:
        1. Loads OHLCV data from database
        2. Loads macro data from database
        3. Uses CanonicalFeatureBuilder to compute features

        Args:
            symbol: Trading symbol
            timestamp: Timestamp for computation
            position: Current position for state features
            bar_idx: Optional bar index (auto-calculated if None)

        Returns:
            Feature array if computed successfully, None otherwise
        """
        if self.feature_builder is None:
            logger.error("CanonicalFeatureBuilder not available for fallback")
            return None

        if not DB_UTILS_AVAILABLE:
            logger.error("Database utilities not available for fallback")
            return None

        try:
            import pandas as pd

            conn = get_db_connection()
            cur = conn.cursor()

            # Load OHLCV data (last 100 bars for indicator warmup)
            cur.execute("""
                SELECT time, open, high, low, close, volume
                FROM usdcop_m5_ohlcv
                WHERE symbol = %s AND time <= %s
                ORDER BY time DESC
                LIMIT 100
            """, (symbol, timestamp))

            rows = cur.fetchall()[::-1]  # Reverse to chronological order

            if len(rows) < 50:  # Need minimum warmup bars
                logger.warning(f"Insufficient OHLCV data: {len(rows)} rows")
                cur.close()
                conn.close()
                return None

            # Create DataFrame
            ohlcv_df = pd.DataFrame(
                rows,
                columns=['time', 'open', 'high', 'low', 'close', 'volume']
            )

            # Load macro data
            cur.execute("""
                SELECT
                    fecha as date,
                    fxrt_index_dxy_usa_d_dxy as dxy,
                    volt_vix_usa_d_vix as vix,
                    comm_oil_brent_glb_d_brent as brent,
                    finc_bond_yield10y_usa_d_ust10y as treasury_10y,
                    fxrt_spot_usdmxn_mex_d_usdmxn as usdmxn,
                    0.0 as embi
                FROM macro_indicators_daily
                WHERE fecha <= %s
                ORDER BY fecha DESC
                LIMIT 30
            """, (timestamp.date(),))

            macro_rows = cur.fetchall()
            cur.close()
            conn.close()

            # Create macro DataFrame
            macro_df = None
            if macro_rows:
                macro_df = pd.DataFrame(
                    macro_rows[::-1],
                    columns=['date', 'dxy', 'vix', 'brent', 'treasury_10y', 'usdmxn', 'embi']
                )

            # Calculate bar index
            if bar_idx is None:
                bar_idx = len(ohlcv_df) - 1

            # Ensure bar_idx is valid
            if bar_idx < 50:
                bar_idx = len(ohlcv_df) - 1

            # Build observation using CanonicalFeatureBuilder
            observation = self.feature_builder.build_observation(
                ohlcv=ohlcv_df,
                macro=macro_df,
                position=position,
                bar_idx=bar_idx,
                timestamp=pd.Timestamp(timestamp),
            )

            logger.info(
                f"Computed fallback features: shape={observation.shape}, "
                f"bar_idx={bar_idx}"
            )

            return observation

        except Exception as e:
            logger.error(f"Error in fallback feature calculation: {e}")
            return None

    def _send_l1_health_alert(self):
        """
        Send alert about L1 pipeline health issues.

        Called when consecutive fallbacks exceed threshold, indicating
        the L1 feature pipeline may be unhealthy.
        """
        alert_message = (
            f"CRITICAL: L1 Feature Pipeline Health Alert\n"
            f"{'=' * 50}\n"
            f"Consecutive fallbacks: {self.consecutive_fallbacks}\n"
            f"Alert threshold: {self.alert_threshold}\n"
            f"Symbol: {self.symbol}\n"
            f"Timestamp: {datetime.now().isoformat()}\n"
            f"\n"
            f"Action Required:\n"
            f"1. Check L1 DAG (v3.l1_feature_refresh) status\n"
            f"2. Verify database connectivity\n"
            f"3. Check inference_features_5m table\n"
            f"4. Review L1 logs for errors\n"
            f"\n"
            f"Using fallback feature computation until L1 is restored."
        )

        # Log at CRITICAL level
        logger.critical(alert_message)

        # Try to send to alerting system if available
        try:
            # Try Slack notification
            from src.shared.notifications.slack_client import send_alert
            send_alert(
                channel="#trading-alerts",
                title="L1 Feature Pipeline Health Alert",
                message=alert_message,
                severity="critical",
            )
        except Exception as e:
            logger.warning(f"Could not send Slack alert: {e}")

        # Try to log to events table
        try:
            if DB_UTILS_AVAILABLE:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO events.system_events
                    (event_type, severity, source, message, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    'l1_health_alert',
                    'critical',
                    'L5InferenceTask',
                    f'L1 consecutive fallbacks: {self.consecutive_fallbacks}',
                    f'{{"fallbacks": {self.consecutive_fallbacks}, "threshold": {self.alert_threshold}}}',
                ))
                conn.commit()
                cur.close()
                conn.close()
        except Exception as e:
            logger.warning(f"Could not log event to database: {e}")

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get the health status of L1 feature retrieval.

        Returns:
            Dict with health metrics
        """
        return {
            "consecutive_fallbacks": self.consecutive_fallbacks,
            "alert_threshold": self.alert_threshold,
            "alert_sent": self._alert_sent,
            "feature_reader_available": self.feature_reader is not None,
            "feature_builder_available": self.feature_builder is not None,
            "symbol": self.symbol,
        }

    def reset_fallback_counter(self):
        """Reset the fallback counter and alert state."""
        self.consecutive_fallbacks = 0
        self._alert_sent = False
        logger.info("L5InferenceTask fallback counter reset")


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_inference_task: Optional[L5InferenceTask] = None


def get_inference_task() -> L5InferenceTask:
    """Get the singleton L5InferenceTask instance."""
    global _inference_task
    if _inference_task is None:
        _inference_task = L5InferenceTask()
    return _inference_task


def reset_inference_task():
    """Reset the singleton instance (for testing)."""
    global _inference_task
    _inference_task = None


# =============================================================================
# AIRFLOW ENTRY POINT
# =============================================================================

def run_l5_inference(**context) -> Dict[str, Any]:
    """
    Airflow PythonOperator entry point for L5 inference.

    This function:
    1. Checks kill switch before proceeding
    2. Gets features using L5InferenceTask
    3. Returns status for XCom

    Args:
        **context: Airflow task context

    Returns:
        Dict with execution status and metadata

    Usage in DAG:
        task = PythonOperator(
            task_id='run_inference',
            python_callable=run_l5_inference,
            provide_context=True,
        )
    """
    logger.info("=" * 60)
    logger.info("L5 INFERENCE TASK STARTED")
    logger.info("=" * 60)

    # Check kill switch
    if TRADING_FLAGS_AVAILABLE:
        try:
            flags = get_trading_flags()
            allowed, reason = flags.is_inference_allowed()

            if not allowed:
                logger.warning(f"Inference blocked: {reason}")
                return {
                    "status": "BLOCKED",
                    "reason": reason,
                    "kill_switch_active": flags.kill_switch,
                }
        except Exception as e:
            logger.warning(f"Could not check trading flags: {e}")
    else:
        logger.warning("TradingFlags not available - proceeding without kill switch check")

    # Get execution parameters
    execution_date = context.get('execution_date') or datetime.utcnow()
    if hasattr(execution_date, 'datetime'):
        execution_date = execution_date.datetime()

    symbol = context.get('params', {}).get('symbol', DEFAULT_SYMBOL)

    # Get inference task instance
    task = get_inference_task()

    # Get features
    features = task.get_features_for_inference(
        symbol=symbol,
        timestamp=execution_date,
    )

    if features is None:
        logger.error("Failed to get features for inference")
        return {
            "status": "ERROR",
            "reason": "Feature retrieval failed",
            "health_status": task.get_health_status(),
        }

    # Push features to XCom for downstream tasks
    ti = context.get('ti')
    if ti:
        ti.xcom_push(key='features', value=features.tolist())
        ti.xcom_push(key='feature_shape', value=features.shape)
        ti.xcom_push(key='fallback_used', value=task.consecutive_fallbacks > 0)

    logger.info(
        f"Features retrieved successfully: shape={features.shape}, "
        f"fallback_used={task.consecutive_fallbacks > 0}"
    )

    return {
        "status": "SUCCESS",
        "feature_shape": features.shape,
        "fallback_used": task.consecutive_fallbacks > 0,
        "health_status": task.get_health_status(),
    }
