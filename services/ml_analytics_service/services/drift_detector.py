"""
Drift Detector Service
======================
Detect data and concept drift in model features.

Author: Pedro @ Lean Tech Solutions
Created: 2025-12-17
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
from database.postgres_client import PostgresClient
from config import METRICS_CONFIG

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detect data and concept drift in model features"""

    # Feature columns to monitor
    FEATURE_COLUMNS = [
        'log_ret_5m', 'log_ret_1h', 'log_ret_4h',
        'rsi_9', 'atr_pct', 'adx_14',
        'dxy_z', 'dxy_change_1d', 'vix_z', 'embi_z',
        'brent_change_1d', 'rate_spread'
    ]

    def __init__(self, db_client: PostgresClient):
        self.db = db_client
        self.warning_threshold = METRICS_CONFIG.drift_warning_threshold
        self.critical_threshold = METRICS_CONFIG.drift_critical_threshold

    def detect_drift(
        self,
        model_id: str,
        window_hours: int = 24,
        baseline_days: int = 7
    ) -> Dict[str, Any]:
        """
        Detect data and concept drift.

        Args:
            model_id: Model identifier
            window_hours: Recent window to compare (hours)
            baseline_days: Baseline period for comparison (days)

        Returns:
            Drift detection results
        """
        # Get recent data (window)
        end_time = datetime.utcnow()
        window_start = end_time - timedelta(hours=window_hours)

        recent_data = self.db.get_inference_data(
            model_id=model_id,
            start_time=window_start.isoformat(),
            end_time=end_time.isoformat(),
            limit=5000
        )

        # Get baseline data
        baseline_end = window_start
        baseline_start = baseline_end - timedelta(days=baseline_days)

        baseline_data = self.db.get_inference_data(
            model_id=model_id,
            start_time=baseline_start.isoformat(),
            end_time=baseline_end.isoformat(),
            limit=5000
        )

        if not recent_data or not baseline_data:
            return {
                'error': 'Insufficient data for drift detection',
                'model_id': model_id,
                'data_drift_score': None,
                'concept_drift_score': None,
                'status': 'unknown'
            }

        # Calculate data drift (feature distribution shift)
        data_drift_results = self._calculate_data_drift(recent_data, baseline_data)

        # Calculate concept drift (prediction accuracy shift)
        concept_drift_results = self._calculate_concept_drift(recent_data, baseline_data)

        # Determine overall status
        max_drift = max(
            data_drift_results['overall_score'],
            concept_drift_results['score']
        )

        if max_drift >= self.critical_threshold:
            status = 'critical'
        elif max_drift >= self.warning_threshold:
            status = 'warning'
        else:
            status = 'healthy'

        return {
            'model_id': model_id,
            'last_check': datetime.utcnow(),
            'window_hours': window_hours,
            'baseline_days': baseline_days,
            'data_drift_score': data_drift_results['overall_score'],
            'concept_drift_score': concept_drift_results['score'],
            'status': status,
            'features_drifted': data_drift_results['drifted_features'],
            'feature_details': data_drift_results['feature_scores'],
            'concept_details': concept_drift_results
        }

    def _calculate_data_drift(
        self,
        recent_data: List[Dict],
        baseline_data: List[Dict]
    ) -> Dict[str, Any]:
        """
        Calculate data drift using Kolmogorov-Smirnov test.

        Returns:
            Data drift results with per-feature scores
        """
        feature_scores = {}
        drifted_features = []

        for feature in self.FEATURE_COLUMNS:
            # Extract feature values
            recent_values = [
                d.get(feature) for d in recent_data
                if d.get(feature) is not None
            ]
            baseline_values = [
                d.get(feature) for d in baseline_data
                if d.get(feature) is not None
            ]

            if len(recent_values) < 10 or len(baseline_values) < 10:
                continue

            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.ks_2samp(recent_values, baseline_values)

            # KS statistic is the drift score (0-1)
            feature_scores[feature] = {
                'ks_statistic': float(ks_stat),
                'p_value': float(p_value),
                'drift_detected': ks_stat > self.warning_threshold,
                'recent_mean': float(np.mean(recent_values)),
                'baseline_mean': float(np.mean(baseline_values)),
                'recent_std': float(np.std(recent_values)),
                'baseline_std': float(np.std(baseline_values))
            }

            if ks_stat > self.warning_threshold:
                drifted_features.append(feature)

        # Overall drift score (average of significant drifts)
        if feature_scores:
            overall_score = np.mean([
                s['ks_statistic'] for s in feature_scores.values()
            ])
        else:
            overall_score = 0.0

        return {
            'overall_score': float(overall_score),
            'feature_scores': feature_scores,
            'drifted_features': drifted_features,
            'total_features_checked': len(feature_scores)
        }

    def _calculate_concept_drift(
        self,
        recent_data: List[Dict],
        baseline_data: List[Dict]
    ) -> Dict[str, Any]:
        """
        Calculate concept drift (prediction accuracy change).

        Returns:
            Concept drift results
        """
        # Calculate accuracy for recent window
        recent_rewards = [d.get('reward', 0) or 0 for d in recent_data]
        recent_correct = sum(1 for r in recent_rewards if r > 0)
        recent_accuracy = recent_correct / len(recent_rewards) if recent_rewards else 0

        # Calculate accuracy for baseline
        baseline_rewards = [d.get('reward', 0) or 0 for d in baseline_data]
        baseline_correct = sum(1 for r in baseline_rewards if r > 0)
        baseline_accuracy = baseline_correct / len(baseline_rewards) if baseline_rewards else 0

        # Concept drift score (absolute difference in accuracy)
        drift_score = abs(recent_accuracy - baseline_accuracy)

        return {
            'score': float(drift_score),
            'recent_accuracy': float(recent_accuracy),
            'baseline_accuracy': float(baseline_accuracy),
            'accuracy_change': float(recent_accuracy - baseline_accuracy),
            'drift_detected': drift_score > self.warning_threshold
        }

    def get_drift_by_feature(
        self,
        model_id: str
    ) -> Dict[str, Any]:
        """
        Get drift detection results grouped by feature.

        Args:
            model_id: Model identifier

        Returns:
            Drift results by feature
        """
        drift_results = self.detect_drift(model_id)

        if 'error' in drift_results:
            return drift_results

        # Reorganize by feature
        features = []
        for feature_name, stats in drift_results.get('feature_details', {}).items():
            features.append({
                'feature': feature_name,
                'drift_score': stats['ks_statistic'],
                'p_value': stats['p_value'],
                'drift_detected': stats['drift_detected'],
                'recent_mean': stats['recent_mean'],
                'baseline_mean': stats['baseline_mean'],
                'mean_change': stats['recent_mean'] - stats['baseline_mean'],
                'recent_std': stats['recent_std'],
                'baseline_std': stats['baseline_std']
            })

        # Sort by drift score
        features.sort(key=lambda x: x['drift_score'], reverse=True)

        return {
            'model_id': model_id,
            'last_check': drift_results['last_check'],
            'overall_status': drift_results['status'],
            'features': features,
            'total_features': len(features),
            'drifted_count': len(drift_results['features_drifted'])
        }
