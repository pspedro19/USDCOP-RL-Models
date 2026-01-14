"""
Data Drift Monitor
==================

Monitors feature distribution drift between training and production data.
Uses Evidently AI for statistical drift detection.

Key metrics:
- Feature-level drift (KS test, PSI)
- Dataset-level drift
- Target drift (if applicable)

Alerts when significant drift is detected, indicating potential model degradation.
"""

import os
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.metrics import (
        DataDriftTable,
        DatasetDriftMetric,
        ColumnDriftMetric,
    )
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logging.warning("evidently not installed. Install with: pip install evidently")

try:
    import redis
except ImportError:
    redis = None

from mlops.config import MLOpsConfig, get_config
from mlops.action_collapse_detector import (
    ActionCollapseDetector,
    ActionCollapseConfig,
    ActionCollapseResult,
)

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """Result of drift detection."""
    timestamp: str
    drift_detected: bool
    drift_share: float  # Proportion of features with drift
    dataset_drift: bool
    drifted_features: List[str]
    feature_scores: Dict[str, float]
    reference_size: int
    current_size: int
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "drift_detected": self.drift_detected,
            "drift_share": self.drift_share,
            "dataset_drift": self.dataset_drift,
            "drifted_features": self.drifted_features,
            "feature_scores": self.feature_scores,
            "reference_size": self.reference_size,
            "current_size": self.current_size,
            "message": self.message,
        }


class DriftMonitor:
    """
    Data drift monitoring using Evidently AI.

    Usage:
        # Initialize with reference data (training data)
        monitor = DriftMonitor(reference_df)

        # Check drift periodically (e.g., hourly)
        result = monitor.check_drift(current_df)

        if result.drift_detected:
            send_alert(f"Data drift detected: {result.drift_share:.1%}")
    """

    # Redis keys
    KEY_DRIFT_LOG = "drift:log"
    KEY_DRIFT_LATEST = "drift:latest"
    KEY_REFERENCE_STATS = "drift:reference_stats"

    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        config: Optional[MLOpsConfig] = None,
        redis_client: Optional[redis.Redis] = None,
        feature_columns: Optional[List[str]] = None,
    ):
        """
        Initialize drift monitor.

        Args:
            reference_data: Reference dataset (training data distribution)
            config: MLOps configuration
            redis_client: Redis client for persistence
            feature_columns: List of feature column names to monitor
        """
        self.config = config or get_config()
        self.drift_threshold = self.config.monitoring.drift_threshold

        # Initialize Redis
        if redis_client:
            self.redis = redis_client
        elif redis:
            try:
                self.redis = redis.Redis(
                    host=self.config.redis.host,
                    port=self.config.redis.port,
                    db=self.config.redis.db,
                    password=self.config.redis.password,
                    decode_responses=True,
                )
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                self.redis = None
        else:
            self.redis = None

        self.reference_data = reference_data
        self.feature_columns = feature_columns

        # Compute reference statistics if data provided
        if reference_data is not None:
            self._compute_reference_stats()

    def _compute_reference_stats(self):
        """Compute and store reference data statistics."""
        if self.reference_data is None:
            return

        if self.feature_columns is None:
            self.feature_columns = [
                col for col in self.reference_data.columns
                if self.reference_data[col].dtype in ['float64', 'float32', 'int64', 'int32']
            ]

        stats = {
            "computed_at": datetime.now().isoformat(),
            "num_samples": len(self.reference_data),
            "num_features": len(self.feature_columns),
            "features": {}
        }

        for col in self.feature_columns:
            if col in self.reference_data.columns:
                col_data = self.reference_data[col].dropna()
                stats["features"][col] = {
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "median": float(col_data.median()),
                    "q25": float(col_data.quantile(0.25)),
                    "q75": float(col_data.quantile(0.75)),
                }

        # Store in Redis
        if self.redis:
            self.redis.set(self.KEY_REFERENCE_STATS, json.dumps(stats))
            self.redis.expire(self.KEY_REFERENCE_STATS, 86400 * 30)  # 30 days

        logger.info(f"Reference stats computed: {len(self.feature_columns)} features, {len(self.reference_data)} samples")

    def set_reference_data(self, reference_data: pd.DataFrame, feature_columns: Optional[List[str]] = None):
        """Set or update reference data."""
        self.reference_data = reference_data
        if feature_columns:
            self.feature_columns = feature_columns
        self._compute_reference_stats()

    def check_drift(
        self,
        current_data: pd.DataFrame,
        custom_threshold: Optional[float] = None
    ) -> DriftResult:
        """
        Check for data drift between reference and current data.

        Args:
            current_data: Current production data
            custom_threshold: Custom drift threshold (overrides config)

        Returns:
            DriftResult with drift detection results
        """
        timestamp = datetime.now().isoformat()
        threshold = custom_threshold or self.drift_threshold

        # Validate inputs
        if self.reference_data is None:
            return DriftResult(
                timestamp=timestamp,
                drift_detected=False,
                drift_share=0.0,
                dataset_drift=False,
                drifted_features=[],
                feature_scores={},
                reference_size=0,
                current_size=len(current_data),
                message="No reference data available"
            )

        if len(current_data) < 10:
            return DriftResult(
                timestamp=timestamp,
                drift_detected=False,
                drift_share=0.0,
                dataset_drift=False,
                drifted_features=[],
                feature_scores={},
                reference_size=len(self.reference_data),
                current_size=len(current_data),
                message="Insufficient current data for drift detection"
            )

        # Use Evidently if available
        if EVIDENTLY_AVAILABLE:
            return self._check_drift_evidently(current_data, timestamp, threshold)
        else:
            return self._check_drift_statistical(current_data, timestamp, threshold)

    def _check_drift_evidently(
        self,
        current_data: pd.DataFrame,
        timestamp: str,
        threshold: float
    ) -> DriftResult:
        """Check drift using Evidently AI."""
        try:
            # Select common columns
            common_cols = [
                col for col in self.feature_columns
                if col in current_data.columns and col in self.reference_data.columns
            ]

            if not common_cols:
                return DriftResult(
                    timestamp=timestamp,
                    drift_detected=False,
                    drift_share=0.0,
                    dataset_drift=False,
                    drifted_features=[],
                    feature_scores={},
                    reference_size=len(self.reference_data),
                    current_size=len(current_data),
                    message="No common columns between reference and current data"
                )

            ref_subset = self.reference_data[common_cols].copy()
            cur_subset = current_data[common_cols].copy()

            # Create report
            report = Report(metrics=[
                DatasetDriftMetric(),
                DataDriftTable(),
            ])

            report.run(
                reference_data=ref_subset,
                current_data=cur_subset
            )

            # Extract results
            result_dict = report.as_dict()

            # Parse metrics
            dataset_drift_result = result_dict['metrics'][0]['result']
            drift_table_result = result_dict['metrics'][1]['result']

            drift_share = dataset_drift_result.get('share_of_drifted_columns', 0.0)
            dataset_drift = dataset_drift_result.get('dataset_drift', False)

            # Get drifted features
            drifted_features = []
            feature_scores = {}

            if 'drift_by_columns' in drift_table_result:
                for col, col_data in drift_table_result['drift_by_columns'].items():
                    score = col_data.get('drift_score', 0.0)
                    feature_scores[col] = score
                    if col_data.get('drift_detected', False):
                        drifted_features.append(col)

            drift_detected = drift_share >= threshold

            result = DriftResult(
                timestamp=timestamp,
                drift_detected=drift_detected,
                drift_share=drift_share,
                dataset_drift=dataset_drift,
                drifted_features=drifted_features,
                feature_scores=feature_scores,
                reference_size=len(self.reference_data),
                current_size=len(current_data),
                message=f"Drift share: {drift_share:.1%} (threshold: {threshold:.1%})"
            )

            # Store result
            self._store_result(result)

            return result

        except Exception as e:
            logger.error(f"Evidently drift check failed: {e}")
            return self._check_drift_statistical(current_data, timestamp, threshold)

    def _check_drift_statistical(
        self,
        current_data: pd.DataFrame,
        timestamp: str,
        threshold: float
    ) -> DriftResult:
        """Fallback statistical drift detection using KS test."""
        from scipy import stats

        drifted_features = []
        feature_scores = {}

        common_cols = [
            col for col in self.feature_columns
            if col in current_data.columns and col in self.reference_data.columns
        ]

        for col in common_cols:
            try:
                ref_values = self.reference_data[col].dropna().values
                cur_values = current_data[col].dropna().values

                if len(ref_values) > 0 and len(cur_values) > 0:
                    # Kolmogorov-Smirnov test
                    statistic, p_value = stats.ks_2samp(ref_values, cur_values)

                    feature_scores[col] = statistic

                    # Consider drift if p-value < 0.05
                    if p_value < 0.05:
                        drifted_features.append(col)

            except Exception as e:
                logger.warning(f"KS test failed for {col}: {e}")

        drift_share = len(drifted_features) / len(common_cols) if common_cols else 0.0
        drift_detected = drift_share >= threshold

        result = DriftResult(
            timestamp=timestamp,
            drift_detected=drift_detected,
            drift_share=drift_share,
            dataset_drift=drift_detected,
            drifted_features=drifted_features,
            feature_scores=feature_scores,
            reference_size=len(self.reference_data),
            current_size=len(current_data),
            message=f"Statistical drift share: {drift_share:.1%} (threshold: {threshold:.1%})"
        )

        self._store_result(result)

        return result

    def _store_result(self, result: DriftResult):
        """Store drift result in Redis."""
        if not self.redis:
            return

        try:
            result_json = json.dumps(result.to_dict())

            # Store latest
            self.redis.set(self.KEY_DRIFT_LATEST, result_json)

            # Add to log
            self.redis.lpush(self.KEY_DRIFT_LOG, result_json)
            self.redis.ltrim(self.KEY_DRIFT_LOG, 0, 999)  # Keep last 1000

            # Set alert flag if drift detected
            if result.drift_detected:
                self.redis.setex("drift:alert", 3600, result_json)  # 1 hour alert

        except Exception as e:
            logger.error(f"Failed to store drift result: {e}")

    def get_latest_result(self) -> Optional[DriftResult]:
        """Get the most recent drift check result."""
        if not self.redis:
            return None

        try:
            data = self.redis.get(self.KEY_DRIFT_LATEST)
            if data:
                d = json.loads(data)
                return DriftResult(**d)
        except Exception as e:
            logger.error(f"Failed to get latest result: {e}")

        return None

    def get_drift_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get drift check history."""
        if not self.redis:
            return []

        try:
            history = self.redis.lrange(self.KEY_DRIFT_LOG, 0, limit - 1)
            return [json.loads(h) for h in history]
        except Exception as e:
            logger.error(f"Failed to get drift history: {e}")
            return []

    def get_reference_stats(self) -> Optional[Dict[str, Any]]:
        """Get stored reference data statistics."""
        if not self.redis:
            return None

        try:
            data = self.redis.get(self.KEY_REFERENCE_STATS)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Failed to get reference stats: {e}")

        return None

    def has_active_alert(self) -> bool:
        """Check if there's an active drift alert."""
        if not self.redis:
            return False

        return self.redis.exists("drift:alert") > 0

    def clear_alert(self):
        """Clear the active drift alert."""
        if self.redis:
            self.redis.delete("drift:alert")

    def check_action_drift(
        self,
        action_history: List[str],
        entropy_threshold: float = 0.5,
        dominance_threshold: float = 0.80
    ) -> ActionCollapseResult:
        """
        Check for action distribution drift (mode collapse).

        This monitors if the model is exhibiting mode collapse behavior,
        i.e., always predicting the same action regardless of observations.

        Args:
            action_history: List of recent action strings ("HOLD", "LONG", "SHORT")
            entropy_threshold: Entropy below this indicates collapse (default 0.5)
            dominance_threshold: Single action above this indicates collapse (default 0.80)

        Returns:
            ActionCollapseResult with analysis details

        Example:
            >>> monitor = DriftMonitor(reference_data)
            >>> actions = ["HOLD"] * 100  # Suspicious pattern
            >>> result = monitor.check_action_drift(actions)
            >>> if result.is_collapsed:
            ...     send_alert(f"Mode collapse detected: {result.warning}")
        """
        config = ActionCollapseConfig(
            entropy_threshold=entropy_threshold,
            dominance_threshold=dominance_threshold,
            min_samples=50,
        )
        detector = ActionCollapseDetector(config)
        result = detector.check(action_history)

        # Log warnings
        if result.is_collapsed:
            logger.warning(f"ACTION COLLAPSE DETECTED: {result.warning}")
            if self.redis:
                self.redis.setex(
                    "action_collapse:alert",
                    3600,  # 1 hour
                    json.dumps(result.to_dict())
                )

        return result

    def generate_report(
        self,
        current_data: pd.DataFrame,
        output_path: str = "drift_report.html"
    ) -> str:
        """
        Generate an HTML drift report.

        Args:
            current_data: Current data to compare
            output_path: Path to save HTML report

        Returns:
            Path to generated report
        """
        if not EVIDENTLY_AVAILABLE:
            raise ImportError("Evidently is required for report generation")

        common_cols = [
            col for col in self.feature_columns
            if col in current_data.columns and col in self.reference_data.columns
        ]

        ref_subset = self.reference_data[common_cols].copy()
        cur_subset = current_data[common_cols].copy()

        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ])

        report.run(
            reference_data=ref_subset,
            current_data=cur_subset
        )

        report.save_html(output_path)
        logger.info(f"Drift report saved to {output_path}")

        return output_path


def create_drift_monitor_from_db(
    connection_string: str,
    reference_query: str,
    feature_columns: List[str],
    config: Optional[MLOpsConfig] = None
) -> DriftMonitor:
    """
    Create a drift monitor with reference data from database.

    Args:
        connection_string: Database connection string
        reference_query: SQL query to get reference data
        feature_columns: List of feature columns
        config: Optional MLOps configuration

    Returns:
        Configured DriftMonitor
    """
    import sqlalchemy as sa

    engine = sa.create_engine(connection_string)
    reference_data = pd.read_sql(reference_query, engine)

    return DriftMonitor(
        reference_data=reference_data,
        config=config,
        feature_columns=feature_columns
    )
