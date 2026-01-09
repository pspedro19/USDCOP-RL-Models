"""
ML Analytics Service - Example Usage
=====================================
Example script demonstrating how to use the ML Analytics API.

Usage:
    python example_usage.py
"""

import requests
import json
from datetime import datetime

# Service URL
BASE_URL = "http://localhost:8004"


def print_json(data, title="Response"):
    """Pretty print JSON response"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(json.dumps(data, indent=2, default=str))
    print(f"{'='*60}\n")


def test_health_check():
    """Test service health"""
    print("1. Testing service health...")
    response = requests.get(f"{BASE_URL}/health")
    print_json(response.json(), "Health Check")


def test_metrics_summary():
    """Test metrics summary endpoint"""
    print("2. Testing metrics summary...")
    response = requests.get(f"{BASE_URL}/api/metrics/summary")
    data = response.json()
    print_json(data, "Metrics Summary")

    # Return first model ID if available
    if data.get('success') and data.get('data', {}).get('models'):
        return data['data']['models'][0].get('model_id')
    return None


def test_rolling_metrics(model_id):
    """Test rolling metrics endpoint"""
    print(f"3. Testing rolling metrics for {model_id}...")
    response = requests.get(
        f"{BASE_URL}/api/metrics/rolling",
        params={'model_id': model_id, 'window': '24h'}
    )
    print_json(response.json(), "Rolling Metrics (24h)")


def test_drift_detection(model_id):
    """Test drift detection endpoint"""
    print(f"4. Testing drift detection for {model_id}...")
    response = requests.get(
        f"{BASE_URL}/api/drift/status",
        params={'model_id': model_id}
    )
    print_json(response.json(), "Drift Detection")


def test_drift_features(model_id):
    """Test drift by features endpoint"""
    print(f"5. Testing drift by features for {model_id}...")
    response = requests.get(
        f"{BASE_URL}/api/drift/features",
        params={'model_id': model_id}
    )
    data = response.json()

    # Show only top 5 features
    if data.get('success') and data.get('data', {}).get('features'):
        data['data']['features'] = data['data']['features'][:5]

    print_json(data, "Drift by Features (Top 5)")


def test_prediction_accuracy(model_id):
    """Test prediction accuracy endpoint"""
    print(f"6. Testing prediction accuracy for {model_id}...")
    response = requests.get(
        f"{BASE_URL}/api/predictions/accuracy",
        params={'model_id': model_id, 'window': '24h'}
    )
    print_json(response.json(), "Prediction Accuracy")


def test_prediction_history(model_id):
    """Test prediction history endpoint"""
    print(f"7. Testing prediction history for {model_id}...")
    response = requests.get(
        f"{BASE_URL}/api/predictions/history",
        params={'model_id': model_id, 'page': 1, 'page_size': 5}
    )
    print_json(response.json(), "Prediction History (First 5)")


def test_models_health():
    """Test models health endpoint"""
    print("8. Testing all models health...")
    response = requests.get(f"{BASE_URL}/api/health/models")
    print_json(response.json(), "All Models Health")


def test_performance_trends(model_id):
    """Test performance trends endpoint"""
    print(f"9. Testing performance trends for {model_id}...")
    response = requests.get(
        f"{BASE_URL}/api/performance/trends/{model_id}",
        params={'days': 7}
    )
    data = response.json()

    # Show only summary to keep output manageable
    if data.get('success') and 'hourly_stats' in data.get('data', {}):
        data['data']['hourly_stats'] = f"[{len(data['data']['hourly_stats'])} hourly records]"

    print_json(data, "Performance Trends (7 days)")


def test_model_comparison():
    """Test model comparison endpoint"""
    print("10. Testing model comparison...")
    response = requests.get(
        f"{BASE_URL}/api/performance/comparison",
        params={'window': '24h'}
    )
    print_json(response.json(), "Model Comparison (24h)")


def main():
    """Run all example requests"""
    print("="*60)
    print("ML Analytics Service - Example Usage")
    print("="*60)
    print(f"Service URL: {BASE_URL}")
    print(f"Time: {datetime.now()}")
    print("="*60)

    try:
        # Test 1: Health check
        test_health_check()

        # Test 2: Get metrics summary and extract model ID
        model_id = test_metrics_summary()

        if not model_id:
            print("\n⚠ No models found. Some tests will be skipped.")
            print("Make sure the database has inference data in dw.fact_rl_inference\n")
            return

        print(f"\n📊 Using model: {model_id}")

        # Test 3: Rolling metrics
        test_rolling_metrics(model_id)

        # Test 4: Drift detection
        test_drift_detection(model_id)

        # Test 5: Drift by features
        test_drift_features(model_id)

        # Test 6: Prediction accuracy
        test_prediction_accuracy(model_id)

        # Test 7: Prediction history
        test_prediction_history(model_id)

        # Test 8: All models health
        test_models_health()

        # Test 9: Performance trends
        test_performance_trends(model_id)

        # Test 10: Model comparison
        test_model_comparison()

        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("="*60)

    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to ML Analytics Service")
        print(f"Make sure the service is running on {BASE_URL}")
        print("\nStart the service with:")
        print("  python main.py")
        print("="*60)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("="*60)


if __name__ == "__main__":
    main()
