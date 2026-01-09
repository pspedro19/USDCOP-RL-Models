"""
Performance Analyzer Service
=============================
Analyze overall model performance and health.

Author: Pedro @ Lean Tech Solutions
Created: 2025-12-17
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import numpy as np
from database.postgres_client import PostgresClient

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """Analyze model performance and health status"""

    def __init__(self, db_client: PostgresClient):
        self.db = db_client

    def get_models_health_status(self) -> Dict[str, Any]:
        """
        Get health status for all active models.

        Returns:
            Health status for all models
        """
        # Get models active in last 24 hours
        query = """
            SELECT
                model_id,
                model_version,
                COUNT(*) as prediction_count,
                AVG(CASE WHEN reward > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
                AVG(latency_ms) as avg_latency_ms,
                MAX(timestamp_utc) as last_prediction,
                MIN(timestamp_utc) as first_prediction
            FROM dw.fact_rl_inference
            WHERE timestamp_utc >= NOW() - INTERVAL '24 hours'
            GROUP BY model_id, model_version
            ORDER BY last_prediction DESC
        """

        models = self.db.execute_query(query)

        # Analyze health for each model
        model_health = []
        for model in models:
            health = self._analyze_model_health(model)
            model_health.append(health)

        # Overall summary
        total_models = len(model_health)
        healthy = sum(1 for m in model_health if m['status'] == 'healthy')
        warning = sum(1 for m in model_health if m['status'] == 'warning')
        critical = sum(1 for m in model_health if m['status'] == 'critical')

        return {
            'timestamp': datetime.utcnow(),
            'total_models': total_models,
            'healthy_models': healthy,
            'warning_models': warning,
            'critical_models': critical,
            'models': model_health
        }

    def _analyze_model_health(self, model_stats: Dict) -> Dict[str, Any]:
        """
        Analyze health status for a single model.

        Args:
            model_stats: Model statistics from database

        Returns:
            Health analysis
        """
        model_id = model_stats['model_id']
        model_version = model_stats.get('model_version')
        prediction_count = model_stats['prediction_count']
        win_rate = float(model_stats.get('win_rate') or 0)
        avg_latency = float(model_stats.get('avg_latency_ms') or 0)
        last_prediction = model_stats['last_prediction']

        # Calculate time since last prediction
        if isinstance(last_prediction, str):
            last_prediction = datetime.fromisoformat(last_prediction.replace('Z', '+00:00'))

        time_since_last = (datetime.utcnow() - last_prediction).total_seconds() / 60  # minutes

        # Health checks
        issues = []
        status = 'healthy'

        # Check 1: Win rate
        if win_rate < 0.45:
            issues.append(f"Low win rate: {win_rate:.2%}")
            status = 'critical'
        elif win_rate < 0.50:
            issues.append(f"Below average win rate: {win_rate:.2%}")
            if status != 'critical':
                status = 'warning'

        # Check 2: Latency
        if avg_latency > 100:
            issues.append(f"High latency: {avg_latency:.0f}ms")
            if status == 'healthy':
                status = 'warning'
        elif avg_latency > 200:
            issues.append(f"Critical latency: {avg_latency:.0f}ms")
            status = 'critical'

        # Check 3: Recency
        if time_since_last > 60:  # More than 1 hour
            issues.append(f"No predictions for {time_since_last:.0f} minutes")
            status = 'critical'
        elif time_since_last > 30:  # More than 30 minutes
            issues.append(f"Stale predictions: {time_since_last:.0f} minutes")
            if status == 'healthy':
                status = 'warning'

        # Check 4: Prediction volume
        if prediction_count < 10:
            issues.append(f"Low prediction count: {prediction_count}")
            if status == 'healthy':
                status = 'warning'

        return {
            'model_id': model_id,
            'model_version': model_version,
            'status': status,
            'prediction_count': prediction_count,
            'win_rate': win_rate,
            'avg_latency_ms': avg_latency,
            'last_prediction': last_prediction,
            'minutes_since_last': time_since_last,
            'issues': issues,
            'health_score': self._calculate_health_score(win_rate, avg_latency, time_since_last)
        }

    def _calculate_health_score(
        self,
        win_rate: float,
        latency: float,
        time_since_last: float
    ) -> float:
        """
        Calculate overall health score (0-100).

        Args:
            win_rate: Win rate (0-1)
            latency: Average latency in ms
            time_since_last: Minutes since last prediction

        Returns:
            Health score (0-100)
        """
        # Win rate component (0-40 points)
        win_rate_score = min(40, win_rate * 80)  # 50% = 40 points, >50% = more

        # Latency component (0-30 points)
        if latency <= 50:
            latency_score = 30
        elif latency <= 100:
            latency_score = 20
        elif latency <= 200:
            latency_score = 10
        else:
            latency_score = 0

        # Recency component (0-30 points)
        if time_since_last <= 10:
            recency_score = 30
        elif time_since_last <= 30:
            recency_score = 20
        elif time_since_last <= 60:
            recency_score = 10
        else:
            recency_score = 0

        return win_rate_score + latency_score + recency_score

    def get_performance_trends(
        self,
        model_id: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get performance trends over time.

        Args:
            model_id: Model identifier
            days: Number of days to analyze

        Returns:
            Performance trends
        """
        start_time = datetime.utcnow() - timedelta(days=days)

        query = """
            SELECT
                DATE_TRUNC('hour', timestamp_utc) as hour,
                COUNT(*) as predictions,
                AVG(CASE WHEN reward > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
                AVG(reward) as avg_reward,
                AVG(latency_ms) as avg_latency_ms,
                SUM(reward) as cumulative_reward
            FROM dw.fact_rl_inference
            WHERE model_id = %s
                AND timestamp_utc >= %s
            GROUP BY DATE_TRUNC('hour', timestamp_utc)
            ORDER BY hour
        """

        hourly_stats = self.db.execute_query(
            query,
            (model_id, start_time.isoformat())
        )

        if not hourly_stats:
            return {
                'model_id': model_id,
                'error': 'No data found',
                'days': days
            }

        # Calculate trends
        win_rates = [float(s['win_rate'] or 0) for s in hourly_stats]
        rewards = [float(s['avg_reward'] or 0) for s in hourly_stats]
        latencies = [float(s['avg_latency_ms'] or 0) for s in hourly_stats if s.get('avg_latency_ms')]

        # Calculate moving averages
        window = 24  # 24-hour moving average
        if len(win_rates) >= window:
            win_rate_ma = np.convolve(win_rates, np.ones(window)/window, mode='valid')
        else:
            win_rate_ma = win_rates

        return {
            'model_id': model_id,
            'start_time': start_time,
            'end_time': datetime.utcnow(),
            'days': days,
            'hourly_stats': hourly_stats,
            'summary': {
                'avg_win_rate': float(np.mean(win_rates)),
                'avg_reward': float(np.mean(rewards)),
                'avg_latency_ms': float(np.mean(latencies)) if latencies else None,
                'win_rate_trend': self._calculate_trend(win_rates),
                'reward_trend': self._calculate_trend(rewards),
                'total_predictions': sum(s['predictions'] for s in hourly_stats),
                'best_hour_win_rate': max(win_rates) if win_rates else 0,
                'worst_hour_win_rate': min(win_rates) if win_rates else 0
            }
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """
        Calculate trend direction (improving, declining, stable).

        Args:
            values: Time series values

        Returns:
            Trend direction
        """
        if len(values) < 2:
            return 'stable'

        # Calculate linear regression slope
        x = np.arange(len(values))
        y = np.array(values)

        # Remove NaN values
        mask = ~np.isnan(y)
        if np.sum(mask) < 2:
            return 'stable'

        x = x[mask]
        y = y[mask]

        slope = np.polyfit(x, y, 1)[0]

        if slope > 0.001:
            return 'improving'
        elif slope < -0.001:
            return 'declining'
        else:
            return 'stable'

    def get_model_comparison(
        self,
        window: str = '24h'
    ) -> Dict[str, Any]:
        """
        Compare performance across all models.

        Args:
            window: Time window for comparison

        Returns:
            Model comparison metrics
        """
        # Parse window
        window_map = {
            '1h': timedelta(hours=1),
            '24h': timedelta(hours=24),
            '7d': timedelta(days=7),
            '30d': timedelta(days=30)
        }
        window_delta = window_map.get(window, timedelta(hours=24))
        start_time = datetime.utcnow() - window_delta

        query = """
            SELECT
                model_id,
                model_version,
                COUNT(*) as predictions,
                AVG(CASE WHEN reward > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
                AVG(reward) as avg_reward,
                STDDEV(reward) as std_reward,
                SUM(reward) as total_reward,
                AVG(latency_ms) as avg_latency_ms,
                MAX(ABS(reward)) as max_abs_reward
            FROM dw.fact_rl_inference
            WHERE timestamp_utc >= %s
            GROUP BY model_id, model_version
            ORDER BY win_rate DESC
        """

        models = self.db.execute_query(query, (start_time.isoformat(),))

        # Calculate Sharpe ratios
        for model in models:
            avg_reward = float(model.get('avg_reward') or 0)
            std_reward = float(model.get('std_reward') or 0.0001)  # Avoid division by zero

            # Annualized Sharpe (252 days * 59 bars = 14,868 bars/year)
            sharpe = (avg_reward / std_reward) * np.sqrt(14868) if std_reward > 0 else 0
            model['sharpe_ratio'] = float(sharpe)

        return {
            'window': window,
            'start_time': start_time,
            'end_time': datetime.utcnow(),
            'total_models': len(models),
            'models': models
        }
