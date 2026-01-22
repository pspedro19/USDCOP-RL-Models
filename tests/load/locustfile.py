"""
Load Testing with Locust
========================

Load testing configuration for the USDCOP Inference API.
Tests API performance under various load conditions.

P2: Load Testing

Usage:
    # Run locally with web UI
    locust -f tests/load/locustfile.py --host http://localhost:8000

    # Run headless for CI
    locust -f tests/load/locustfile.py --host http://localhost:8000 \
           --headless -u 100 -r 10 -t 60s --html=report.html

    # Distributed mode (master)
    locust -f tests/load/locustfile.py --master

    # Distributed mode (worker)
    locust -f tests/load/locustfile.py --worker --master-host=localhost

Author: Trading Team
Version: 1.0.0
Date: 2026-01-17
"""

import json
import random
from datetime import datetime, timedelta

from locust import HttpUser, task, between, events, tag
from locust.runners import MasterRunner, WorkerRunner


# =============================================================================
# Test Data Generators
# =============================================================================

def generate_feature_payload() -> dict:
    """Generate realistic feature data for prediction requests."""
    return {
        "features": {
            "log_ret_5m": random.gauss(0, 0.005),
            "log_ret_1h": random.gauss(0, 0.01),
            "log_ret_1d": random.gauss(0, 0.02),
            "rsi_14": random.uniform(20, 80),
            "macd": random.gauss(0, 10),
            "macd_signal": random.gauss(0, 8),
            "bb_upper": random.uniform(4100, 4200),
            "bb_lower": random.uniform(3900, 4000),
            "atr_14": random.uniform(20, 50),
            "dxy": random.uniform(100, 110),
            "vix": random.uniform(12, 25),
            "volume_sma_ratio": random.uniform(0.8, 1.2),
            "spread_bps": random.uniform(1, 5),
            "hour_sin": random.uniform(-1, 1),
            "hour_cos": random.uniform(-1, 1),
        },
        "timestamp": datetime.utcnow().isoformat(),
        "model_id": "ppo_primary",
    }


def generate_backtest_request() -> dict:
    """Generate backtest request payload."""
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=random.randint(7, 30))

    return {
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "model_id": random.choice(["ppo_primary", "ppo_secondary"]),
        "initial_capital": 10000,
    }


# =============================================================================
# User Classes
# =============================================================================

class InferenceAPIUser(HttpUser):
    """
    Simulates a typical API user making inference requests.

    This user performs:
    - Health checks
    - Single predictions
    - Model info queries
    """

    wait_time = between(0.5, 2.0)  # Wait 0.5-2 seconds between tasks
    weight = 10  # Higher weight = more common user type

    def on_start(self):
        """Setup before running tasks."""
        # Verify the API is accessible
        response = self.client.get("/api/v1/health")
        if response.status_code != 200:
            raise Exception("API health check failed")

    @task(10)
    @tag("health")
    def health_check(self):
        """Check API health endpoint."""
        with self.client.get(
            "/api/v1/health",
            name="/api/v1/health",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @task(5)
    @tag("models")
    def list_models(self):
        """List available models."""
        self.client.get("/api/v1/models", name="/api/v1/models")

    @task(3)
    @tag("config")
    def get_config(self):
        """Get current configuration."""
        self.client.get("/api/v1/config", name="/api/v1/config")

    @task(1)
    @tag("info")
    def get_root(self):
        """Get API root info."""
        self.client.get("/", name="/")


class HighFrequencyTrader(HttpUser):
    """
    Simulates a high-frequency trading client.

    Makes rapid prediction requests with minimal wait time.
    """

    wait_time = between(0.1, 0.5)  # Very fast requests
    weight = 3  # Less common but important to test

    @task(20)
    @tag("predict", "critical")
    def make_prediction(self):
        """Request a trade prediction."""
        payload = generate_feature_payload()

        with self.client.post(
            "/api/v1/predict",
            json=payload,
            name="/api/v1/predict",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "action" in data or "prediction" in data:
                    response.success()
                else:
                    response.failure("Invalid prediction response")
            elif response.status_code == 429:
                response.failure("Rate limited")
            else:
                response.failure(f"Prediction failed: {response.status_code}")

    @task(5)
    @tag("health")
    def quick_health(self):
        """Quick health check between predictions."""
        self.client.get("/api/v1/health", name="/api/v1/health [HFT]")


class BacktestUser(HttpUser):
    """
    Simulates a user running backtests.

    Makes longer-running backtest requests.
    """

    wait_time = between(5.0, 15.0)  # Backtests are less frequent
    weight = 1  # Rare but resource-intensive

    @task(1)
    @tag("backtest")
    def run_backtest(self):
        """Run a backtest."""
        payload = generate_backtest_request()

        with self.client.post(
            "/api/v1/backtest",
            json=payload,
            name="/api/v1/backtest",
            timeout=120,  # Backtests can take longer
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "trades" in data or "results" in data:
                    response.success()
                else:
                    response.failure("Invalid backtest response")
            elif response.status_code == 429:
                response.failure("Rate limited during backtest")
            else:
                response.failure(f"Backtest failed: {response.status_code}")


class MixedWorkloadUser(HttpUser):
    """
    Simulates a typical user with mixed workload.

    Combines various API operations.
    """

    wait_time = between(1.0, 3.0)
    weight = 5

    @task(10)
    @tag("health")
    def health_check(self):
        """Regular health checks."""
        self.client.get("/api/v1/health")

    @task(8)
    @tag("predict")
    def predict(self):
        """Make predictions."""
        payload = generate_feature_payload()
        self.client.post("/api/v1/predict", json=payload)

    @task(3)
    @tag("models")
    def check_models(self):
        """Check model status."""
        self.client.get("/api/v1/models")

    @task(2)
    @tag("trades")
    def get_trades(self):
        """Get recent trades."""
        self.client.get("/api/v1/trades")

    @task(1)
    @tag("replay")
    def get_replay(self):
        """Get feature replay."""
        self.client.get("/api/v1/replay/latest")


# =============================================================================
# Event Handlers
# =============================================================================

@events.init.add_listener
def on_locust_init(environment, **kwargs):
    """Initialize Locust environment."""
    if isinstance(environment.runner, MasterRunner):
        print("Running as master node")
    elif isinstance(environment.runner, WorkerRunner):
        print("Running as worker node")
    else:
        print("Running in standalone mode")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Log test start."""
    print(f"Load test starting with {environment.runner.user_count} users")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Log test completion and summary."""
    stats = environment.runner.stats
    print("\n" + "=" * 60)
    print("LOAD TEST SUMMARY")
    print("=" * 60)
    print(f"Total requests: {stats.total.num_requests}")
    print(f"Total failures: {stats.total.num_failures}")
    print(f"Failure rate: {stats.total.fail_ratio:.2%}")
    print(f"Avg response time: {stats.total.avg_response_time:.0f}ms")
    print(f"P50 response time: {stats.total.get_response_time_percentile(0.50):.0f}ms")
    print(f"P95 response time: {stats.total.get_response_time_percentile(0.95):.0f}ms")
    print(f"P99 response time: {stats.total.get_response_time_percentile(0.99):.0f}ms")
    print(f"Requests/s: {stats.total.current_rps:.1f}")
    print("=" * 60)


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, context, exception, **kwargs):
    """Track individual requests for custom metrics."""
    if exception:
        print(f"Request failed: {name} - {exception}")
    elif response_time > 1000:  # Flag slow requests (>1s)
        print(f"Slow request: {name} - {response_time}ms")


# =============================================================================
# Custom Shape (for advanced load patterns)
# =============================================================================

class StepLoadShape:
    """
    Step load shape for gradual ramp-up testing.

    Increases load in steps to find breaking points.
    """

    def __init__(self):
        self.step_duration = 60  # seconds per step
        self.step_users = 10     # users to add per step
        self.max_users = 100     # maximum users
        self.start_time = None

    def tick(self):
        """Return current user count and spawn rate."""
        if self.start_time is None:
            self.start_time = datetime.now()

        run_time = (datetime.now() - self.start_time).total_seconds()
        current_step = int(run_time // self.step_duration)
        target_users = min(self.step_users * (current_step + 1), self.max_users)

        if target_users > self.max_users:
            return None  # Stop the test

        return (target_users, self.step_users)  # (users, spawn_rate)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # This allows running with: python locustfile.py
    import subprocess
    subprocess.run([
        "locust",
        "-f", __file__,
        "--host", "http://localhost:8000",
        "--web-host", "0.0.0.0",
    ])
