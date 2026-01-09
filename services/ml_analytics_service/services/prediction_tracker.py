"""
Prediction Tracker Service
===========================
Track predictions vs actual outcomes.

Author: Pedro @ Lean Tech Solutions
Created: 2025-12-17
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from database.postgres_client import PostgresClient

logger = logging.getLogger(__name__)


class PredictionTracker:
    """Track and analyze model predictions"""

    def __init__(self, db_client: PostgresClient):
        self.db = db_client

    def get_prediction_accuracy(
        self,
        model_id: str,
        window: str = '24h'
    ) -> Dict[str, Any]:
        """
        Calculate prediction accuracy metrics.

        Args:
            model_id: Model identifier
            window: Time window (1h, 24h, 7d, 30d)

        Returns:
            Prediction accuracy metrics
        """
        # Parse window
        window_delta = self._parse_window(window)
        end_time = datetime.utcnow()
        start_time = end_time - window_delta

        # Get predictions
        predictions = self.db.get_inference_data(
            model_id=model_id,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            limit=10000
        )

        if not predictions:
            return {
                'error': 'No predictions found',
                'model_id': model_id,
                'window': window
            }

        # Calculate accuracy metrics
        total = len(predictions)
        correct = sum(1 for p in predictions if (p.get('reward') or 0) > 0)
        incorrect = sum(1 for p in predictions if (p.get('reward') or 0) < 0)
        pending = sum(1 for p in predictions if (p.get('reward') or 0) == 0)

        accuracy = correct / total if total > 0 else 0

        # Calculate by action
        by_action = self._calculate_by_action(predictions)

        # Build confusion matrix (simplified)
        confusion_matrix = self._build_confusion_matrix(predictions)

        return {
            'model_id': model_id,
            'window': window,
            'start_time': start_time,
            'end_time': end_time,
            'total_predictions': total,
            'correct_predictions': correct,
            'incorrect_predictions': incorrect,
            'pending_predictions': pending,
            'accuracy': accuracy,
            'precision': accuracy,  # Simplified
            'recall': accuracy,  # Simplified
            'f1_score': accuracy,  # Simplified
            'by_action': by_action,
            'confusion_matrix': confusion_matrix
        }

    def get_prediction_history(
        self,
        model_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        page: int = 1,
        page_size: int = 100
    ) -> Dict[str, Any]:
        """
        Get historical predictions with outcomes.

        Args:
            model_id: Model identifier
            start_time: Start time (optional)
            end_time: End time (optional)
            page: Page number
            page_size: Results per page

        Returns:
            Prediction history
        """
        if end_time is None:
            end_time = datetime.utcnow()
        if start_time is None:
            start_time = end_time - timedelta(days=7)

        # Get predictions with pagination
        offset = (page - 1) * page_size
        query = """
            SELECT
                inference_id,
                timestamp_utc,
                timestamp_cot,
                model_id,
                model_version,
                action_raw,
                action_discretized,
                confidence,
                close_price,
                position_before,
                position_after,
                reward,
                latency_ms,
                log_ret_5m as actual_return
            FROM dw.fact_rl_inference
            WHERE model_id = %s
                AND timestamp_utc >= %s
                AND timestamp_utc <= %s
            ORDER BY timestamp_utc DESC
            LIMIT %s OFFSET %s
        """

        predictions = self.db.execute_query(
            query,
            (model_id, start_time.isoformat(), end_time.isoformat(), page_size, offset)
        )

        # Get total count
        count_query = """
            SELECT COUNT(*) as total
            FROM dw.fact_rl_inference
            WHERE model_id = %s
                AND timestamp_utc >= %s
                AND timestamp_utc <= %s
        """
        count_result = self.db.execute_single(
            count_query,
            (model_id, start_time.isoformat(), end_time.isoformat())
        )
        total = count_result['total'] if count_result else 0

        # Calculate quick stats
        if predictions:
            confidences = [p.get('confidence') for p in predictions if p.get('confidence')]
            latencies = [p.get('latency_ms') for p in predictions if p.get('latency_ms')]
            rewards = [p.get('reward', 0) or 0 for p in predictions]

            avg_confidence = sum(confidences) / len(confidences) if confidences else None
            avg_latency = sum(latencies) / len(latencies) if latencies else None
            avg_reward = sum(rewards) / len(rewards) if rewards else None
        else:
            avg_confidence = None
            avg_latency = None
            avg_reward = None

        # Mark predictions as profitable
        for pred in predictions:
            pred['was_profitable'] = (pred.get('reward') or 0) > 0

        return {
            'model_id': model_id,
            'start_time': start_time,
            'end_time': end_time,
            'predictions': predictions,
            'total': total,
            'page': page,
            'page_size': page_size,
            'avg_confidence': avg_confidence,
            'avg_latency_ms': avg_latency,
            'avg_reward': avg_reward
        }

    def compare_predictions_vs_actuals(
        self,
        model_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Compare predictions against actual market outcomes.

        Args:
            model_id: Model identifier
            limit: Number of recent predictions to compare

        Returns:
            List of prediction comparisons
        """
        query = """
            SELECT
                timestamp_utc,
                action_discretized as predicted_action,
                close_price,
                reward,
                log_ret_5m as actual_return,
                position_after - position_before as position_change
            FROM dw.fact_rl_inference
            WHERE model_id = %s
            ORDER BY timestamp_utc DESC
            LIMIT %s
        """

        predictions = self.db.execute_query(query, (model_id, limit))

        comparisons = []
        for pred in predictions:
            # Determine optimal action in hindsight
            actual_return = pred.get('actual_return') or 0
            if actual_return > 0:
                optimal_action = 'LONG'
            elif actual_return < 0:
                optimal_action = 'SHORT'
            else:
                optimal_action = 'HOLD'

            # Calculate error
            predicted_return = pred.get('reward') or 0
            error = abs(predicted_return - actual_return)

            comparisons.append({
                'timestamp': pred['timestamp_utc'],
                'predicted_action': pred['predicted_action'],
                'actual_optimal_action': optimal_action,
                'predicted_return': predicted_return,
                'actual_return': actual_return,
                'error': error,
                'close_price': pred['close_price'],
                'was_correct': pred['predicted_action'] == optimal_action
            })

        return comparisons

    def _calculate_by_action(self, predictions: List[Dict]) -> List[Dict]:
        """Calculate statistics by action type"""
        actions = {}

        for pred in predictions:
            action = pred.get('action_discretized', 'UNKNOWN')
            if action not in actions:
                actions[action] = {
                    'action': action,
                    'count': 0,
                    'confidences': [],
                    'rewards': [],
                    'successes': 0
                }

            actions[action]['count'] += 1

            confidence = pred.get('confidence')
            if confidence is not None:
                actions[action]['confidences'].append(confidence)

            reward = pred.get('reward', 0) or 0
            actions[action]['rewards'].append(reward)

            if reward > 0:
                actions[action]['successes'] += 1

        # Calculate averages
        by_action = []
        for stats in actions.values():
            by_action.append({
                'action': stats['action'],
                'count': stats['count'],
                'avg_confidence': (
                    sum(stats['confidences']) / len(stats['confidences'])
                    if stats['confidences'] else None
                ),
                'avg_reward': (
                    sum(stats['rewards']) / len(stats['rewards'])
                    if stats['rewards'] else None
                ),
                'success_rate': stats['successes'] / stats['count'] if stats['count'] > 0 else 0
            })

        return by_action

    def _build_confusion_matrix(self, predictions: List[Dict]) -> Dict[str, Dict[str, int]]:
        """Build confusion matrix for predictions"""
        matrix = {
            'LONG': {'LONG': 0, 'SHORT': 0, 'HOLD': 0},
            'SHORT': {'LONG': 0, 'SHORT': 0, 'HOLD': 0},
            'HOLD': {'LONG': 0, 'SHORT': 0, 'HOLD': 0}
        }

        for pred in predictions:
            predicted = pred.get('action_discretized', 'HOLD')
            reward = pred.get('reward', 0) or 0

            # Determine actual (simplified)
            if reward > 0:
                actual = predicted  # Correct prediction
            else:
                # Incorrect - assign to different action
                if predicted == 'LONG':
                    actual = 'SHORT'
                elif predicted == 'SHORT':
                    actual = 'LONG'
                else:
                    actual = 'HOLD'

            if predicted in matrix and actual in matrix[predicted]:
                matrix[predicted][actual] += 1

        return matrix

    def _parse_window(self, window: str) -> timedelta:
        """Parse window string to timedelta"""
        window_map = {
            '1h': timedelta(hours=1),
            '24h': timedelta(hours=24),
            '7d': timedelta(days=7),
            '30d': timedelta(days=30)
        }
        return window_map.get(window, timedelta(hours=24))
