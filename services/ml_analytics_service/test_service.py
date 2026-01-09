"""
ML Analytics Service - Test Script
===================================
Quick test script to verify service functionality.

Usage:
    python test_service.py
"""

import os
import sys
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.postgres_client import PostgresClient
from services.metrics_calculator import MetricsCalculator
from services.drift_detector import DriftDetector
from services.prediction_tracker import PredictionTracker
from services.performance_analyzer import PerformanceAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_database_connection():
    """Test database connection"""
    logger.info("Testing database connection...")
    try:
        db = PostgresClient(min_connections=1, max_connections=2)
        if db.test_connection():
            logger.info("✓ Database connection successful")
            return db
        else:
            logger.error("✗ Database connection failed")
            return None
    except Exception as e:
        logger.error(f"✗ Database connection error: {e}")
        return None


def test_data_availability(db: PostgresClient):
    """Test if inference data is available"""
    logger.info("Testing data availability...")
    try:
        # Check for inference data
        query = """
            SELECT COUNT(*) as count
            FROM dw.fact_rl_inference
            WHERE timestamp_utc >= NOW() - INTERVAL '7 days'
        """
        result = db.execute_single(query)
        count = result['count'] if result else 0

        if count > 0:
            logger.info(f"✓ Found {count} inference records in last 7 days")
            return True
        else:
            logger.warning("✗ No inference data found in last 7 days")
            return False
    except Exception as e:
        logger.error(f"✗ Error checking data availability: {e}")
        return False


def test_metrics_calculator(db: PostgresClient):
    """Test metrics calculator"""
    logger.info("Testing metrics calculator...")
    try:
        calc = MetricsCalculator(db)
        summary = calc.get_metrics_summary()

        if summary and summary.get('total_models', 0) > 0:
            logger.info(f"✓ Metrics calculator working - found {summary['total_models']} models")
            return True
        else:
            logger.warning("✗ No models found in metrics summary")
            return False
    except Exception as e:
        logger.error(f"✗ Metrics calculator error: {e}")
        return False


def test_drift_detector(db: PostgresClient, model_id: str):
    """Test drift detector"""
    logger.info("Testing drift detector...")
    try:
        detector = DriftDetector(db)
        drift = detector.detect_drift(model_id)

        if 'error' not in drift:
            logger.info(f"✓ Drift detector working - status: {drift.get('status', 'unknown')}")
            return True
        else:
            logger.warning(f"✗ Drift detector error: {drift.get('error')}")
            return False
    except Exception as e:
        logger.error(f"✗ Drift detector error: {e}")
        return False


def test_prediction_tracker(db: PostgresClient, model_id: str):
    """Test prediction tracker"""
    logger.info("Testing prediction tracker...")
    try:
        tracker = PredictionTracker(db)
        accuracy = tracker.get_prediction_accuracy(model_id, '24h')

        if 'error' not in accuracy:
            logger.info(f"✓ Prediction tracker working - {accuracy.get('total_predictions', 0)} predictions")
            return True
        else:
            logger.warning(f"✗ Prediction tracker error: {accuracy.get('error')}")
            return False
    except Exception as e:
        logger.error(f"✗ Prediction tracker error: {e}")
        return False


def test_performance_analyzer(db: PostgresClient):
    """Test performance analyzer"""
    logger.info("Testing performance analyzer...")
    try:
        analyzer = PerformanceAnalyzer(db)
        health = analyzer.get_models_health_status()

        if health and health.get('total_models', 0) > 0:
            logger.info(f"✓ Performance analyzer working - {health['total_models']} models")
            return True
        else:
            logger.warning("✗ No models found in health status")
            return False
    except Exception as e:
        logger.error(f"✗ Performance analyzer error: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("ML Analytics Service - Test Suite")
    logger.info("=" * 60)

    results = {}

    # Test 1: Database connection
    db = test_database_connection()
    results['database'] = db is not None

    if not db:
        logger.error("Cannot proceed without database connection")
        return

    # Test 2: Data availability
    results['data'] = test_data_availability(db)

    if not results['data']:
        logger.warning("Limited tests can be performed without data")

    # Test 3: Metrics calculator
    results['metrics'] = test_metrics_calculator(db)

    # Get a sample model ID for testing
    model_id = None
    if results['data']:
        try:
            query = """
                SELECT DISTINCT model_id
                FROM dw.fact_rl_inference
                WHERE timestamp_utc >= NOW() - INTERVAL '7 days'
                LIMIT 1
            """
            result = db.execute_single(query)
            model_id = result['model_id'] if result else None
        except Exception as e:
            logger.error(f"Error getting sample model ID: {e}")

    # Test 4: Drift detector
    if model_id:
        results['drift'] = test_drift_detector(db, model_id)
    else:
        logger.warning("Skipping drift detector test - no model ID")
        results['drift'] = False

    # Test 5: Prediction tracker
    if model_id:
        results['prediction'] = test_prediction_tracker(db, model_id)
    else:
        logger.warning("Skipping prediction tracker test - no model ID")
        results['prediction'] = False

    # Test 6: Performance analyzer
    results['performance'] = test_performance_analyzer(db)

    # Summary
    logger.info("=" * 60)
    logger.info("Test Results Summary")
    logger.info("=" * 60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{test_name:20s}: {status}")

    passed = sum(results.values())
    total = len(results)
    logger.info("=" * 60)
    logger.info(f"Total: {passed}/{total} tests passed")
    logger.info("=" * 60)

    # Cleanup
    db.close()

    # Exit code
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
