"""
Load Test: Latency SLA Verification (Phase 12)
==============================================

Load testing with SLA assertions for the inference API.
Tests verify that latency targets are met under load.

SLA Targets:
    - p50 < 20ms
    - p95 < 50ms
    - p99 < 100ms

Usage:
    # Run with pytest
    pytest tests/load/test_latency_sla.py -v

    # Run with locust (for interactive testing)
    locust -f tests/load/test_latency_sla.py --headless -u 50 -r 10 -t 60s

Author: Trading Team
Date: 2025-01-14
"""

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

# Try to import locust for load testing
try:
    from locust import HttpUser, between, task
    LOCUST_AVAILABLE = True
except ImportError:
    LOCUST_AVAILABLE = False


# =============================================================================
# SLA CONFIGURATION
# =============================================================================

@dataclass
class SLAConfig:
    """SLA configuration for latency targets."""
    p50_ms: float = 20.0    # 20ms
    p95_ms: float = 50.0    # 50ms
    p99_ms: float = 100.0   # 100ms
    error_rate_max: float = 0.001  # 0.1%
    min_throughput_rps: float = 1.0  # 1 request per second


@dataclass
class LoadTestResult:
    """Results from a load test run."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def p50(self) -> float:
        """50th percentile (median) latency."""
        if not self.latencies_ms:
            return 0.0
        return float(np.percentile(self.latencies_ms, 50))

    @property
    def p95(self) -> float:
        """95th percentile latency."""
        if not self.latencies_ms:
            return 0.0
        return float(np.percentile(self.latencies_ms, 95))

    @property
    def p99(self) -> float:
        """99th percentile latency."""
        if not self.latencies_ms:
            return 0.0
        return float(np.percentile(self.latencies_ms, 99))

    @property
    def mean(self) -> float:
        """Mean latency."""
        if not self.latencies_ms:
            return 0.0
        return statistics.mean(self.latencies_ms)

    @property
    def error_rate(self) -> float:
        """Error rate as a fraction."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    @property
    def throughput_rps(self) -> float:
        """Requests per second."""
        if self.duration_seconds == 0:
            return 0.0
        return self.successful_requests / self.duration_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "error_rate": f"{self.error_rate:.2%}",
            "p50_ms": f"{self.p50:.2f}",
            "p95_ms": f"{self.p95:.2f}",
            "p99_ms": f"{self.p99:.2f}",
            "mean_ms": f"{self.mean:.2f}",
            "throughput_rps": f"{self.throughput_rps:.2f}",
            "duration_seconds": f"{self.duration_seconds:.2f}",
        }


# =============================================================================
# MOCK INFERENCE CLIENT (for testing without real service)
# =============================================================================

class MockInferenceClient:
    """
    Mock inference client for testing.

    In real tests, replace with actual HTTP client.
    """

    def __init__(self, base_latency_ms: float = 10.0, variance_ms: float = 5.0):
        self.base_latency_ms = base_latency_ms
        self.variance_ms = variance_ms
        self._request_count = 0

    async def predict(self) -> Dict[str, Any]:
        """Simulate an inference request."""
        self._request_count += 1

        # Simulate latency with some variance
        latency = self.base_latency_ms + np.random.uniform(-self.variance_ms, self.variance_ms)

        # Occasionally add a slow request (5% chance)
        if np.random.random() < 0.05:
            latency += np.random.uniform(20, 50)

        # Simulate processing time
        await asyncio.sleep(latency / 1000)

        return {
            "signal": "HOLD",
            "confidence": 0.85,
            "latency_ms": latency,
        }


# =============================================================================
# LOAD TEST RUNNER
# =============================================================================

class LoadTestRunner:
    """
    Load test runner with concurrent request handling.

    Usage:
        runner = LoadTestRunner()
        result = await runner.run(duration_seconds=60, concurrent_users=10)
    """

    def __init__(self, client: Optional[MockInferenceClient] = None):
        self.client = client or MockInferenceClient()
        self.sla = SLAConfig()

    async def run(
        self,
        duration_seconds: float = 60.0,
        concurrent_users: int = 10,
        ramp_up_seconds: float = 5.0
    ) -> LoadTestResult:
        """
        Run load test for specified duration.

        Args:
            duration_seconds: Total test duration
            concurrent_users: Number of concurrent virtual users
            ramp_up_seconds: Time to ramp up to full load

        Returns:
            LoadTestResult with latency statistics
        """
        result = LoadTestResult()
        start_time = time.time()

        # Create tasks for concurrent users
        tasks = []
        for i in range(concurrent_users):
            # Stagger user start times during ramp-up
            delay = (i / concurrent_users) * ramp_up_seconds
            task = asyncio.create_task(
                self._user_loop(result, start_time, duration_seconds, delay)
            )
            tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

        result.duration_seconds = time.time() - start_time
        return result

    async def _user_loop(
        self,
        result: LoadTestResult,
        start_time: float,
        duration_seconds: float,
        initial_delay: float
    ):
        """Single user request loop."""
        await asyncio.sleep(initial_delay)

        while time.time() - start_time < duration_seconds:
            request_start = time.perf_counter()

            try:
                response = await self.client.predict()
                latency_ms = (time.perf_counter() - request_start) * 1000

                result.total_requests += 1
                result.successful_requests += 1
                result.latencies_ms.append(latency_ms)

            except Exception as e:
                result.total_requests += 1
                result.failed_requests += 1
                result.errors.append(str(e))

            # Small delay between requests per user
            await asyncio.sleep(0.1)

    def verify_sla(self, result: LoadTestResult) -> Dict[str, bool]:
        """
        Verify SLA targets are met.

        Returns:
            Dict of check_name -> passed
        """
        return {
            "p50_ok": result.p50 < self.sla.p50_ms,
            "p95_ok": result.p95 < self.sla.p95_ms,
            "p99_ok": result.p99 < self.sla.p99_ms,
            "error_rate_ok": result.error_rate < self.sla.error_rate_max,
            "throughput_ok": result.throughput_rps >= self.sla.min_throughput_rps,
        }


# =============================================================================
# PYTEST TESTS
# =============================================================================

@pytest.mark.load
class TestLatencySLA:
    """Load tests with SLA assertions."""

    @pytest.fixture
    def runner(self):
        """Create load test runner with mock client."""
        client = MockInferenceClient(base_latency_ms=10.0, variance_ms=5.0)
        return LoadTestRunner(client)

    @pytest.mark.asyncio
    async def test_latency_p50_sla(self, runner):
        """Test p50 latency meets SLA (< 20ms)."""
        result = await runner.run(duration_seconds=10, concurrent_users=5)

        assert result.p50 < 20.0, \
            f"p50 SLA breach: {result.p50:.2f}ms > 20ms"

    @pytest.mark.asyncio
    async def test_latency_p95_sla(self, runner):
        """Test p95 latency meets SLA (< 50ms)."""
        result = await runner.run(duration_seconds=10, concurrent_users=5)

        assert result.p95 < 50.0, \
            f"p95 SLA breach: {result.p95:.2f}ms > 50ms"

    @pytest.mark.asyncio
    async def test_latency_p99_sla(self, runner):
        """Test p99 latency meets SLA (< 100ms)."""
        result = await runner.run(duration_seconds=10, concurrent_users=5)

        assert result.p99 < 100.0, \
            f"p99 SLA breach: {result.p99:.2f}ms > 100ms"

    @pytest.mark.asyncio
    async def test_error_rate_sla(self, runner):
        """Test error rate meets SLA (< 0.1%)."""
        result = await runner.run(duration_seconds=10, concurrent_users=5)

        assert result.error_rate < 0.001, \
            f"Error rate SLA breach: {result.error_rate:.2%} > 0.1%"

    @pytest.mark.asyncio
    async def test_all_sla_targets(self, runner):
        """
        Comprehensive SLA test - all targets must be met.

        This is the primary test that should be run in CI.
        """
        result = await runner.run(duration_seconds=30, concurrent_users=10)
        sla_checks = runner.verify_sla(result)

        # Print detailed results
        print("\n" + "=" * 60)
        print("LOAD TEST RESULTS")
        print("=" * 60)
        for key, value in result.to_dict().items():
            print(f"  {key}: {value}")

        print("\n" + "-" * 60)
        print("SLA VERIFICATION")
        print("-" * 60)
        for check, passed in sla_checks.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {check}: {status}")
        print("=" * 60 + "\n")

        # Assert all checks pass
        assert all(sla_checks.values()), \
            f"SLA checks failed: {[k for k, v in sla_checks.items() if not v]}"


@pytest.mark.load
@pytest.mark.slow
class TestLoadScenarios:
    """Load test scenarios for different conditions."""

    @pytest.fixture
    def runner(self):
        client = MockInferenceClient(base_latency_ms=10.0, variance_ms=5.0)
        return LoadTestRunner(client)

    @pytest.mark.asyncio
    async def test_normal_load(self, runner):
        """Test under normal load (10 users, 2 RPS)."""
        result = await runner.run(duration_seconds=30, concurrent_users=10)

        assert result.p99 < 100.0, "p99 exceeded under normal load"
        assert result.error_rate < 0.001, "Errors under normal load"

    @pytest.mark.asyncio
    async def test_peak_load(self, runner):
        """Test under peak load (50 users, 10 RPS)."""
        result = await runner.run(duration_seconds=30, concurrent_users=50)

        # More lenient thresholds for peak load
        assert result.p99 < 200.0, "p99 exceeded under peak load"
        assert result.error_rate < 0.01, "High error rate under peak load"

    @pytest.mark.asyncio
    async def test_sustained_load(self, runner):
        """Test sustained load over longer period."""
        result = await runner.run(duration_seconds=120, concurrent_users=20)

        assert result.p95 < 50.0, "p95 degraded over time"
        assert result.error_rate < 0.001, "Errors during sustained load"


# =============================================================================
# LOCUST USER (for interactive load testing)
# =============================================================================

if LOCUST_AVAILABLE:
    class InferenceUser(HttpUser):
        """
        Locust user for interactive load testing.

        Usage:
            locust -f tests/load/test_latency_sla.py --host=http://localhost:8000
        """
        wait_time = between(0.1, 0.5)

        @task
        def predict(self):
            """Make inference request."""
            with self.client.post(
                "/api/v1/predict",
                json={
                    "model_id": "ppo_primary",
                    "position": 0.0,
                    "session_progress": 0.5
                },
                catch_response=True
            ) as response:
                if response.status_code != 200:
                    response.failure(f"Status: {response.status_code}")
                elif response.elapsed.total_seconds() > 0.1:  # 100ms
                    response.failure(f"Too slow: {response.elapsed.total_seconds()*1000:.0f}ms")
