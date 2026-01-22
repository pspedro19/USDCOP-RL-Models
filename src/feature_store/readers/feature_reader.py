"""
Feature Reader - Read Pre-computed Features from L1 Pipeline
============================================================
This module provides the FeatureReader class for reading pre-computed features
from the inference_features_5m table populated by the L1 pipeline.

Instead of recalculating features at inference time (L5), this reader fetches
the already computed features from L1, ensuring:
1. Perfect parity between feature calculation (done once in L1)
2. Reduced latency at inference time
3. Single source of truth for feature values

Architecture:
    L0 (OHLCV/Macro) -> L1 (Feature Calculation) -> inference_features_5m
                                                            |
                                                            v
    L5 (Inference) <----- FeatureReader <-----------------+

Usage:
    from src.feature_store.readers import FeatureReader

    reader = FeatureReader()
    result = reader.get_latest_features("USD/COP", max_age_minutes=10)

    if result:
        observation = result.observation  # np.ndarray ready for model
        print(f"Features age: {result.age_minutes:.1f} minutes")

Author: Trading Team
Version: 1.0.0
Created: 2026-01-17
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureResult:
    """
    Result of a feature read operation.

    Contains the observation array ready for model inference along with
    metadata about the features for validation and auditing.

    Attributes:
        observation: numpy array of features in FEATURE_ORDER
        timestamp: timestamp of the feature bar
        age_minutes: how old the features are (for staleness checks)
        norm_stats_hash: hash of normalization stats used (for parity validation)
        raw_features: dictionary of feature name -> value (for debugging)
        source: source of the features (default: "l1_pipeline")
    """
    observation: np.ndarray
    timestamp: datetime
    age_minutes: float
    norm_stats_hash: str
    raw_features: Dict[str, float]
    source: str = "l1_pipeline"

    def __post_init__(self):
        """Validate the observation array."""
        if self.observation is not None:
            if not isinstance(self.observation, np.ndarray):
                self.observation = np.array(self.observation, dtype=np.float32)
            elif self.observation.dtype != np.float32:
                self.observation = self.observation.astype(np.float32)

    def is_valid(self) -> bool:
        """Check if the feature result is valid for inference."""
        if self.observation is None:
            return False
        if len(self.observation) == 0:
            return False
        if np.isnan(self.observation).any():
            return False
        if np.isinf(self.observation).any():
            return False
        return True

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "age_minutes": self.age_minutes,
            "norm_stats_hash": self.norm_stats_hash,
            "raw_features": self.raw_features,
            "source": self.source,
            "observation_shape": list(self.observation.shape) if self.observation is not None else None,
            "is_valid": self.is_valid(),
        }


class FeatureReaderError(Exception):
    """Base exception for FeatureReader errors."""
    pass


class FeatureNotFoundError(FeatureReaderError):
    """Raised when no features are found for the given criteria."""
    pass


class StaleFeatureError(FeatureReaderError):
    """Raised when features are too old."""
    pass


class FeatureOrderMismatchError(FeatureReaderError):
    """Raised when feature order doesn't match expected contract."""
    pass


class FeatureReader:
    """
    Read pre-computed features from the L1 pipeline.

    This class fetches features from the inference_features_5m table
    populated by L1, avoiding redundant feature calculation in L5.

    The reader ensures:
    1. Features are in the correct FEATURE_ORDER
    2. Features are not stale (configurable max age)
    3. All required features are present

    Example:
        reader = FeatureReader()

        # Get latest features for inference
        result = reader.get_latest_features(
            symbol="USD/COP",
            timestamp=datetime.utcnow(),
            max_age_minutes=10
        )

        if result and result.is_valid():
            action, _ = model.predict(result.observation)
        else:
            logger.warning("No valid features available")
    """

    # Default feature order from feature_store.core
    DEFAULT_FEATURE_ORDER: Tuple[str, ...] = (
        "log_ret_5m", "log_ret_1h", "log_ret_4h",
        "rsi_9", "atr_pct", "adx_14",
        "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
        "brent_change_1d", "rate_spread", "usdmxn_change_1d",
        "position", "time_normalized"
    )

    def __init__(
        self,
        feature_order: Optional[Tuple[str, ...]] = None,
        default_position: float = 0.0,
        default_time_normalized: float = 0.5,
    ):
        """
        Initialize the FeatureReader.

        Args:
            feature_order: Feature order tuple. If None, imports from core module.
            default_position: Default position value if not in database (default: 0.0)
            default_time_normalized: Default time_normalized value (default: 0.5)
        """
        # Import FEATURE_ORDER from SSOT (REQUIRED - no fallback)
        if feature_order is None:
            from src.core.contracts import FEATURE_ORDER
            self._feature_order = FEATURE_ORDER
            logger.info(f"Using FEATURE_ORDER from SSOT: {len(FEATURE_ORDER)} features")
        else:
            self._feature_order = feature_order

        self._observation_dim = len(self._feature_order)
        self._default_position = default_position
        self._default_time_normalized = default_time_normalized
        self._norm_stats_hash = self._compute_norm_stats_hash()

        logger.info(
            f"FeatureReader initialized: {self._observation_dim} features, "
            f"norm_stats_hash={self._norm_stats_hash[:12]}..."
        )

    @property
    def feature_order(self) -> Tuple[str, ...]:
        """Get the feature order tuple."""
        return self._feature_order

    @property
    def observation_dim(self) -> int:
        """Get the observation dimension."""
        return self._observation_dim

    def _compute_norm_stats_hash(self) -> str:
        """Compute hash of normalization stats for parity validation."""
        try:
            from src.feature_store.core import NORM_STATS_PATH
            from pathlib import Path
            import json

            # Find norm stats file
            path = Path(NORM_STATS_PATH)
            if not path.is_absolute():
                project_root = Path(__file__).parent.parent.parent.parent
                path = project_root / NORM_STATS_PATH

            if path.exists():
                with open(path) as f:
                    stats = json.load(f)
                stats_str = json.dumps(stats, sort_keys=True)
                return hashlib.md5(stats_str.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"Could not compute norm_stats_hash: {e}")

        return "unknown"

    def get_latest_features(
        self,
        symbol: str,
        timestamp: Optional[datetime] = None,
        max_age_minutes: float = 10.0,
        position: Optional[float] = None,
        time_normalized: Optional[float] = None,
    ) -> Optional[FeatureResult]:
        """
        Get the latest features for inference.

        Queries the inference_features_5m table for the most recent features
        that are not older than max_age_minutes.

        Args:
            symbol: Trading symbol (e.g., "USD/COP")
            timestamp: Reference timestamp (default: now)
            max_age_minutes: Maximum age of features in minutes (default: 10)
            position: Override position value (default: use db value or 0.0)
            time_normalized: Override time_normalized (default: compute from timestamp)

        Returns:
            FeatureResult if valid features found, None otherwise

        Raises:
            FeatureNotFoundError: If no features found
            StaleFeatureError: If features are too old
            FeatureOrderMismatchError: If features don't match expected order
        """
        from src.database import get_psycopg2_connection

        if timestamp is None:
            timestamp = datetime.utcnow()

        with get_psycopg2_connection() as conn:
            cur = conn.cursor()

            try:
                # Query latest features from inference_features_5m
                # Note: The table doesn't have a symbol column, assuming single symbol
                cur.execute("""
                    SELECT
                        time,
                        log_ret_5m, log_ret_1h, log_ret_4h,
                        rsi_9, atr_pct, adx_14,
                        dxy_z, dxy_change_1d, vix_z, embi_z,
                        brent_change_1d, rate_spread, usdmxn_change_1d,
                        position, time_normalized,
                        EXTRACT(EPOCH FROM (%s - time)) / 60.0 as age_minutes
                    FROM inference_features_5m
                    WHERE time <= %s
                    ORDER BY time DESC
                    LIMIT 1
                """, (timestamp, timestamp))

                row = cur.fetchone()

                if row is None:
                    logger.warning(f"No features found for {symbol} at {timestamp}")
                    return None

                feature_time = row[0]
                age_minutes = float(row[-1]) if row[-1] is not None else 0.0

                # Check staleness
                if age_minutes > max_age_minutes:
                    logger.warning(
                        f"Features too old: {age_minutes:.1f} min > {max_age_minutes} min"
                    )
                    return None

                # Build raw features dict
                raw_features = {
                    "log_ret_5m": self._safe_float(row[1]),
                    "log_ret_1h": self._safe_float(row[2]),
                    "log_ret_4h": self._safe_float(row[3]),
                    "rsi_9": self._safe_float(row[4]),
                    "atr_pct": self._safe_float(row[5]),
                    "adx_14": self._safe_float(row[6]),
                    "dxy_z": self._safe_float(row[7]),
                    "dxy_change_1d": self._safe_float(row[8]),
                    "vix_z": self._safe_float(row[9]),
                    "embi_z": self._safe_float(row[10]),
                    "brent_change_1d": self._safe_float(row[11]),
                    "rate_spread": self._safe_float(row[12]),
                    "usdmxn_change_1d": self._safe_float(row[13]),
                    # State features - use override or db value
                    "position": position if position is not None else self._safe_float(row[14], self._default_position),
                    "time_normalized": time_normalized if time_normalized is not None else self._safe_float(row[15], self._default_time_normalized),
                }

                # Validate feature order
                if not self._validate_feature_order(raw_features):
                    logger.error("Feature order validation failed")
                    return None

                # Build observation array in correct order
                observation = self._build_observation(raw_features)

                return FeatureResult(
                    observation=observation,
                    timestamp=feature_time,
                    age_minutes=age_minutes,
                    norm_stats_hash=self._norm_stats_hash,
                    raw_features=raw_features,
                    source="l1_pipeline",
                )

            finally:
                cur.close()

    def _validate_feature_order(self, features_dict: Dict[str, float]) -> bool:
        """
        Validate that all required features are present.

        Args:
            features_dict: Dictionary of feature name -> value

        Returns:
            True if all required features are present, False otherwise
        """
        missing_features = []
        for feature_name in self._feature_order:
            if feature_name not in features_dict:
                missing_features.append(feature_name)

        if missing_features:
            logger.error(f"Missing features: {missing_features}")
            return False

        return True

    def _build_observation(self, raw_features: Dict[str, float]) -> np.ndarray:
        """
        Build observation array from raw features in FEATURE_ORDER.

        Args:
            raw_features: Dictionary of feature name -> value

        Returns:
            numpy array of shape (observation_dim,)
        """
        observation = np.zeros(self._observation_dim, dtype=np.float32)

        for idx, feature_name in enumerate(self._feature_order):
            value = raw_features.get(feature_name, 0.0)
            # Handle NaN/Inf
            if value is None or np.isnan(value) or np.isinf(value):
                value = 0.0
            observation[idx] = value

        return observation

    def _safe_float(self, value, default: float = 0.0) -> float:
        """Safely convert value to float."""
        if value is None:
            return default
        try:
            f = float(value)
            if np.isnan(f) or np.isinf(f):
                return default
            return f
        except (TypeError, ValueError):
            return default

    def get_features_history(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 1000,
    ) -> List[FeatureResult]:
        """
        Get historical features for a time range.

        Useful for backtesting or feature analysis.

        Args:
            symbol: Trading symbol (e.g., "USD/COP")
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum number of records to return (default: 1000)

        Returns:
            List of FeatureResult objects, ordered by timestamp ascending
        """
        from src.database import get_psycopg2_connection

        results = []

        with get_psycopg2_connection() as conn:
            cur = conn.cursor()

            try:
                cur.execute("""
                    SELECT
                        time,
                        log_ret_5m, log_ret_1h, log_ret_4h,
                        rsi_9, atr_pct, adx_14,
                        dxy_z, dxy_change_1d, vix_z, embi_z,
                        brent_change_1d, rate_spread, usdmxn_change_1d,
                        position, time_normalized
                    FROM inference_features_5m
                    WHERE time >= %s AND time <= %s
                    ORDER BY time ASC
                    LIMIT %s
                """, (start_time, end_time, limit))

                rows = cur.fetchall()
                reference_time = end_time

                for row in rows:
                    feature_time = row[0]
                    age_minutes = (reference_time - feature_time).total_seconds() / 60.0

                    raw_features = {
                        "log_ret_5m": self._safe_float(row[1]),
                        "log_ret_1h": self._safe_float(row[2]),
                        "log_ret_4h": self._safe_float(row[3]),
                        "rsi_9": self._safe_float(row[4]),
                        "atr_pct": self._safe_float(row[5]),
                        "adx_14": self._safe_float(row[6]),
                        "dxy_z": self._safe_float(row[7]),
                        "dxy_change_1d": self._safe_float(row[8]),
                        "vix_z": self._safe_float(row[9]),
                        "embi_z": self._safe_float(row[10]),
                        "brent_change_1d": self._safe_float(row[11]),
                        "rate_spread": self._safe_float(row[12]),
                        "usdmxn_change_1d": self._safe_float(row[13]),
                        "position": self._safe_float(row[14], self._default_position),
                        "time_normalized": self._safe_float(row[15], self._default_time_normalized),
                    }

                    observation = self._build_observation(raw_features)

                    results.append(FeatureResult(
                        observation=observation,
                        timestamp=feature_time,
                        age_minutes=age_minutes,
                        norm_stats_hash=self._norm_stats_hash,
                        raw_features=raw_features,
                        source="l1_pipeline",
                    ))

                logger.info(
                    f"Retrieved {len(results)} feature records from {start_time} to {end_time}"
                )

            finally:
                cur.close()

        return results

    def check_feature_freshness(
        self,
        symbol: str,
        max_age_minutes: float = 10.0,
    ) -> Tuple[bool, float, Optional[datetime]]:
        """
        Check if features are fresh enough for inference.

        Args:
            symbol: Trading symbol
            max_age_minutes: Maximum acceptable age in minutes

        Returns:
            Tuple of (is_fresh, age_minutes, last_timestamp)
        """
        from src.database import get_psycopg2_connection

        with get_psycopg2_connection() as conn:
            cur = conn.cursor()

            try:
                cur.execute("""
                    SELECT
                        time,
                        EXTRACT(EPOCH FROM (NOW() - time)) / 60.0 as age_minutes
                    FROM inference_features_5m
                    ORDER BY time DESC
                    LIMIT 1
                """)

                row = cur.fetchone()

                if row is None:
                    return False, float('inf'), None

                last_time = row[0]
                age_minutes = float(row[1])

                is_fresh = age_minutes <= max_age_minutes

                return is_fresh, age_minutes, last_time

            finally:
                cur.close()
