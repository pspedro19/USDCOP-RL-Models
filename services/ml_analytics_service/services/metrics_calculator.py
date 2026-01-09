"""
Metrics Calculator Service
===========================
Calculate rolling metrics for model performance.

Author: Pedro @ Lean Tech Solutions
Created: 2025-12-17
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import numpy as np
from database.postgres_client import PostgresClient

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate performance metrics for RL models"""

    def __init__(self, db_client: PostgresClient):
        self.db = db_client

    def calculate_rolling_metrics(
        self,
        model_id: str,
        window: str = '24h'
    ) -> Dict[str, Any]:
        """
        Calculate rolling window metrics.

        Args:
            model_id: Model identifier
            window: Time window (1h, 24h, 7d, 30d)

        Returns:
            Dictionary with metrics
        """
        # Convert window to timedelta
        window_delta = self._parse_window(window)
        end_time = datetime.utcnow()
        start_time = end_time - window_delta

        # Get inference data
        inferences = self.db.get_inference_data(
            model_id=model_id,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            limit=10000
        )

        if not inferences:
            return {
                'model_id': model_id,
                'window': window,
                'start_time': start_time,
                'end_time': end_time,
                'metrics': {},
                'error': 'No data found for this window'
            }

        # Calculate metrics
        metrics = self._calculate_metrics(inferences)

        return {
            'model_id': model_id,
            'window': window,
            'start_time': start_time,
            'end_time': end_time,
            'data_points': len(inferences),
            'metrics': metrics,
            'predictions': self._calculate_prediction_stats(inferences)
        }

    def _calculate_metrics(self, inferences: List[Dict]) -> Dict[str, float]:
        """Calculate performance metrics from inference data"""
        if not inferences:
            return {}

        rewards = [inf.get('reward', 0) or 0 for inf in inferences]
        latencies = [inf.get('latency_ms') for inf in inferences if inf.get('latency_ms')]
        positions = [inf.get('position_after', 0) or 0 for inf in inferences]

        # Basic metrics
        total_return = sum(rewards)
        avg_return = np.mean(rewards) if rewards else 0
        std_return = np.std(rewards) if len(rewards) > 1 else 0

        # Win rate
        positive_rewards = [r for r in rewards if r > 0]
        win_rate = len(positive_rewards) / len(rewards) if rewards else 0

        # Sharpe ratio (annualized, assuming 5-min bars)
        # 252 trading days * 59 bars per day = 14,868 bars per year
        if std_return > 0:
            sharpe_ratio = (avg_return / std_return) * np.sqrt(14868)
        else:
            sharpe_ratio = 0

        # Profit factor
        gross_profit = sum(positive_rewards) if positive_rewards else 0
        negative_rewards = [abs(r) for r in rewards if r < 0]
        gross_loss = sum(negative_rewards) if negative_rewards else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Maximum drawdown
        cumulative_returns = np.cumsum(rewards)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = running_max - cumulative_returns
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        # MAE and MSE (for returns)
        mae = np.mean(np.abs(rewards)) if rewards else 0
        mse = np.mean(np.square(rewards)) if rewards else 0

        # Accuracy (correct direction)
        correct_predictions = sum(1 for r in rewards if r > 0)
        accuracy = correct_predictions / len(rewards) if rewards else 0

        return {
            'accuracy': float(accuracy),
            'precision': float(win_rate),  # Approximation
            'recall': float(win_rate),  # Approximation
            'f1_score': float(win_rate),  # Approximation
            'mse': float(mse),
            'mae': float(mae),
            'sharpe_ratio': float(sharpe_ratio),
            'profit_factor': float(profit_factor),
            'win_rate': float(win_rate),
            'max_drawdown': float(max_drawdown),
            'avg_latency_ms': float(np.mean(latencies)) if latencies else None,
            'total_return': float(total_return),
            'avg_return': float(avg_return),
            'std_return': float(std_return)
        }

    def _calculate_prediction_stats(self, inferences: List[Dict]) -> Dict[str, Any]:
        """Calculate prediction statistics"""
        total = len(inferences)
        correct = sum(1 for inf in inferences if (inf.get('reward') or 0) > 0)

        # Count by action
        by_action = {}
        for inf in inferences:
            action = inf.get('action_discretized', 'UNKNOWN')
            by_action[action] = by_action.get(action, 0) + 1

        return {
            'total': total,
            'correct': correct,
            'by_action': by_action
        }

    def _parse_window(self, window: str) -> timedelta:
        """Parse window string to timedelta"""
        window_map = {
            '1h': timedelta(hours=1),
            '24h': timedelta(hours=24),
            '7d': timedelta(days=7),
            '30d': timedelta(days=30)
        }
        return window_map.get(window, timedelta(hours=24))

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary metrics for all models"""
        # Get unique model IDs
        query = """
            SELECT DISTINCT model_id, model_version
            FROM dw.fact_rl_inference
            WHERE timestamp_utc >= NOW() - INTERVAL '7 days'
            ORDER BY model_id
        """
        models = self.db.execute_query(query)

        # Calculate metrics for each model
        model_metrics = []
        for model in models:
            model_id = model['model_id']
            metrics = self.calculate_rolling_metrics(model_id, '24h')
            model_metrics.append(metrics)

        # Aggregate metrics
        if model_metrics:
            accuracies = [m['metrics'].get('accuracy', 0) for m in model_metrics if m.get('metrics')]
            sharpes = [m['metrics'].get('sharpe_ratio', 0) for m in model_metrics if m.get('metrics')]
            win_rates = [m['metrics'].get('win_rate', 0) for m in model_metrics if m.get('metrics')]

            aggregate = {
                'accuracy': float(np.mean(accuracies)) if accuracies else None,
                'sharpe_ratio': float(np.mean(sharpes)) if sharpes else None,
                'win_rate': float(np.mean(win_rates)) if win_rates else None
            }
        else:
            aggregate = {}

        return {
            'timestamp': datetime.utcnow(),
            'total_models': len(models),
            'active_models': len(model_metrics),
            'models': model_metrics,
            'aggregate': aggregate
        }
