"""
Chaos Testing Framework
=======================

Chaos engineering tests to verify system resilience under failure conditions.
Simulates various failure scenarios to ensure graceful degradation.

P2: Chaos Testing

Tests:
- Network latency injection
- Service failure simulation
- Database connection failures
- Memory pressure
- CPU pressure
- Disk I/O failures

Author: Trading Team
Version: 1.0.0
Date: 2026-01-17
"""

import asyncio
import logging
import os
import random
import signal
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import pytest
import requests

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ChaosConfig:
    """Configuration for chaos tests."""
    # Service URLs
    api_url: str = "http://localhost:8000"
    postgres_url: str = "postgresql://localhost:5432/test"
    redis_url: str = "redis://localhost:6379"

    # Test parameters
    test_duration_seconds: int = 30
    recovery_timeout_seconds: int = 60
    health_check_interval: float = 1.0

    # Failure parameters
    latency_ms: int = 500
    failure_probability: float = 0.3
    memory_pressure_mb: int = 512
    cpu_cores_to_stress: int = 2

    # Docker container names
    api_container: str = "usdcop-backtest-api"
    postgres_container: str = "usdcop-postgres-timescale"
    redis_container: str = "usdcop-redis"


@dataclass
class ChaosTestResult:
    """Result of a chaos test."""
    test_name: str
    passed: bool
    duration_seconds: float
    requests_made: int = 0
    requests_succeeded: int = 0
    requests_failed: int = 0
    recovery_time_seconds: Optional[float] = None
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate request success rate."""
        if self.requests_made == 0:
            return 1.0
        return self.requests_succeeded / self.requests_made


# =============================================================================
# Base Chaos Test
# =============================================================================

class ChaosTest(ABC):
    """Base class for chaos tests."""

    def __init__(self, config: ChaosConfig):
        self.config = config
        self.result: Optional[ChaosTestResult] = None

    @abstractmethod
    def inject_failure(self) -> None:
        """Inject the failure condition."""
        pass

    @abstractmethod
    def remove_failure(self) -> None:
        """Remove the failure condition."""
        pass

    def check_health(self) -> bool:
        """Check if the API is healthy."""
        try:
            response = requests.get(
                f"{self.config.api_url}/api/v1/health",
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False

    def make_request(self) -> bool:
        """Make a test request to the API."""
        try:
            response = requests.get(
                f"{self.config.api_url}/api/v1/health",
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False

    def wait_for_recovery(self) -> float:
        """Wait for service to recover and return recovery time."""
        start_time = time.time()
        timeout = self.config.recovery_timeout_seconds

        while time.time() - start_time < timeout:
            if self.check_health():
                return time.time() - start_time
            time.sleep(self.config.health_check_interval)

        raise TimeoutError(f"Service did not recover within {timeout}s")

    def run(self) -> ChaosTestResult:
        """Run the chaos test."""
        test_name = self.__class__.__name__
        start_time = time.time()
        requests_made = 0
        requests_succeeded = 0

        logger.info(f"Starting chaos test: {test_name}")

        try:
            # Verify service is healthy before starting
            if not self.check_health():
                return ChaosTestResult(
                    test_name=test_name,
                    passed=False,
                    duration_seconds=time.time() - start_time,
                    error_message="Service not healthy before test",
                )

            # Inject failure
            logger.info(f"Injecting failure...")
            self.inject_failure()

            # Make requests during failure
            test_end = time.time() + self.config.test_duration_seconds
            while time.time() < test_end:
                requests_made += 1
                if self.make_request():
                    requests_succeeded += 1
                time.sleep(0.1)

            # Remove failure
            logger.info(f"Removing failure...")
            self.remove_failure()

            # Wait for recovery
            recovery_time = self.wait_for_recovery()
            logger.info(f"Service recovered in {recovery_time:.1f}s")

            # Verify service is fully recovered
            final_health = self.check_health()

            result = ChaosTestResult(
                test_name=test_name,
                passed=final_health,
                duration_seconds=time.time() - start_time,
                requests_made=requests_made,
                requests_succeeded=requests_succeeded,
                requests_failed=requests_made - requests_succeeded,
                recovery_time_seconds=recovery_time,
            )

        except Exception as e:
            logger.error(f"Chaos test failed: {e}")
            self.remove_failure()  # Clean up on failure

            result = ChaosTestResult(
                test_name=test_name,
                passed=False,
                duration_seconds=time.time() - start_time,
                requests_made=requests_made,
                requests_succeeded=requests_succeeded,
                requests_failed=requests_made - requests_succeeded,
                error_message=str(e),
            )

        self.result = result
        return result


# =============================================================================
# Specific Chaos Tests
# =============================================================================

class NetworkLatencyTest(ChaosTest):
    """
    Inject network latency using tc (traffic control).

    Simulates slow network conditions.
    """

    def inject_failure(self) -> None:
        """Add network latency to the API container."""
        # Using Docker to inject latency via tc
        cmd = [
            "docker", "exec", self.config.api_container,
            "tc", "qdisc", "add", "dev", "eth0", "root", "netem",
            "delay", f"{self.config.latency_ms}ms"
        ]
        try:
            subprocess.run(cmd, check=False, capture_output=True)
            logger.info(f"Injected {self.config.latency_ms}ms latency")
        except Exception as e:
            logger.warning(f"Could not inject latency (may require tc): {e}")

    def remove_failure(self) -> None:
        """Remove network latency."""
        cmd = [
            "docker", "exec", self.config.api_container,
            "tc", "qdisc", "del", "dev", "eth0", "root"
        ]
        try:
            subprocess.run(cmd, check=False, capture_output=True)
        except Exception:
            pass


class ContainerPauseTest(ChaosTest):
    """
    Pause a Docker container to simulate service freeze.

    Tests how the system handles unresponsive services.
    """

    def inject_failure(self) -> None:
        """Pause the API container."""
        cmd = ["docker", "pause", self.config.api_container]
        subprocess.run(cmd, check=False, capture_output=True)
        logger.info(f"Paused container: {self.config.api_container}")

    def remove_failure(self) -> None:
        """Unpause the API container."""
        cmd = ["docker", "unpause", self.config.api_container]
        subprocess.run(cmd, check=False, capture_output=True)
        logger.info(f"Unpaused container: {self.config.api_container}")


class DatabaseConnectionTest(ChaosTest):
    """
    Kill database connections to test connection pool resilience.

    Simulates database connection storms.
    """

    def inject_failure(self) -> None:
        """Kill database connections."""
        # This would typically use pg_terminate_backend
        cmd = [
            "docker", "exec", self.config.postgres_container,
            "psql", "-U", "admin", "-c",
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'usdcop' AND pid <> pg_backend_pid();"
        ]
        try:
            subprocess.run(cmd, check=False, capture_output=True)
            logger.info("Terminated database connections")
        except Exception as e:
            logger.warning(f"Could not terminate DB connections: {e}")

    def remove_failure(self) -> None:
        """No explicit removal needed - connections will be re-established."""
        pass


class RedisConnectionTest(ChaosTest):
    """
    Restart Redis to test cache resilience.

    Tests graceful degradation when cache is unavailable.
    """

    def inject_failure(self) -> None:
        """Stop Redis container."""
        cmd = ["docker", "stop", self.config.redis_container]
        subprocess.run(cmd, check=False, capture_output=True)
        logger.info("Stopped Redis container")

    def remove_failure(self) -> None:
        """Start Redis container."""
        cmd = ["docker", "start", self.config.redis_container]
        subprocess.run(cmd, check=False, capture_output=True)
        logger.info("Started Redis container")


class MemoryPressureTest(ChaosTest):
    """
    Apply memory pressure to test OOM handling.

    Allocates memory to stress the container.
    """

    def __init__(self, config: ChaosConfig):
        super().__init__(config)
        self._stress_proc = None

    def inject_failure(self) -> None:
        """Apply memory pressure using stress tool."""
        cmd = [
            "docker", "exec", "-d", self.config.api_container,
            "stress", "--vm", "1",
            "--vm-bytes", f"{self.config.memory_pressure_mb}M",
            "--timeout", f"{self.config.test_duration_seconds}s"
        ]
        try:
            subprocess.run(cmd, check=False, capture_output=True)
            logger.info(f"Applied {self.config.memory_pressure_mb}MB memory pressure")
        except Exception as e:
            logger.warning(f"Could not apply memory pressure: {e}")

    def remove_failure(self) -> None:
        """Memory pressure will timeout automatically."""
        # Kill any remaining stress processes
        cmd = [
            "docker", "exec", self.config.api_container,
            "pkill", "-f", "stress"
        ]
        subprocess.run(cmd, check=False, capture_output=True)


class CPUPressureTest(ChaosTest):
    """
    Apply CPU pressure to test performance under load.

    Stresses CPU cores to simulate compute-heavy conditions.
    """

    def inject_failure(self) -> None:
        """Apply CPU pressure using stress tool."""
        cmd = [
            "docker", "exec", "-d", self.config.api_container,
            "stress", "--cpu", str(self.config.cpu_cores_to_stress),
            "--timeout", f"{self.config.test_duration_seconds}s"
        ]
        try:
            subprocess.run(cmd, check=False, capture_output=True)
            logger.info(f"Applied CPU pressure on {self.config.cpu_cores_to_stress} cores")
        except Exception as e:
            logger.warning(f"Could not apply CPU pressure: {e}")

    def remove_failure(self) -> None:
        """CPU pressure will timeout automatically."""
        cmd = [
            "docker", "exec", self.config.api_container,
            "pkill", "-f", "stress"
        ]
        subprocess.run(cmd, check=False, capture_output=True)


# =============================================================================
# Chaos Test Runner
# =============================================================================

class ChaosTestRunner:
    """
    Runs a suite of chaos tests and generates reports.

    Usage:
        runner = ChaosTestRunner(config)
        results = runner.run_all()
        runner.print_report()
    """

    def __init__(self, config: Optional[ChaosConfig] = None):
        self.config = config or ChaosConfig()
        self.results: List[ChaosTestResult] = []
        self.tests: List[ChaosTest] = [
            NetworkLatencyTest(self.config),
            ContainerPauseTest(self.config),
            DatabaseConnectionTest(self.config),
            RedisConnectionTest(self.config),
            MemoryPressureTest(self.config),
            CPUPressureTest(self.config),
        ]

    def run_all(self) -> List[ChaosTestResult]:
        """Run all chaos tests."""
        self.results = []

        for test in self.tests:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running: {test.__class__.__name__}")
            logger.info(f"{'='*60}")

            result = test.run()
            self.results.append(result)

            # Wait between tests
            logger.info("Waiting for system to stabilize...")
            time.sleep(10)

        return self.results

    def run_test(self, test_name: str) -> Optional[ChaosTestResult]:
        """Run a specific test by name."""
        for test in self.tests:
            if test.__class__.__name__ == test_name:
                result = test.run()
                self.results.append(result)
                return result
        return None

    def print_report(self) -> None:
        """Print a summary report of all test results."""
        print("\n" + "=" * 60)
        print("CHAOS TEST REPORT")
        print("=" * 60)

        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed

        print(f"\nTotal tests: {len(self.results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")

        print("\n" + "-" * 60)
        print("DETAILED RESULTS")
        print("-" * 60)

        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            print(f"\n{result.test_name}: {status}")
            print(f"  Duration: {result.duration_seconds:.1f}s")
            print(f"  Requests: {result.requests_made} made, "
                  f"{result.requests_succeeded} succeeded ({result.success_rate:.1%})")
            if result.recovery_time_seconds:
                print(f"  Recovery time: {result.recovery_time_seconds:.1f}s")
            if result.error_message:
                print(f"  Error: {result.error_message}")

        print("\n" + "=" * 60)


# =============================================================================
# Pytest Integration
# =============================================================================

@pytest.fixture
def chaos_config():
    """Provide chaos test configuration."""
    return ChaosConfig(
        api_url=os.getenv("API_URL", "http://localhost:8000"),
        test_duration_seconds=10,  # Shorter for CI
        recovery_timeout_seconds=30,
    )


@pytest.mark.chaos
class TestChaos:
    """Pytest-based chaos tests."""

    def test_network_latency(self, chaos_config):
        """Test system handles network latency gracefully."""
        test = NetworkLatencyTest(chaos_config)
        result = test.run()
        assert result.passed, f"Network latency test failed: {result.error_message}"
        assert result.success_rate >= 0.5, "Success rate too low during latency"

    def test_container_pause(self, chaos_config):
        """Test system recovers from container pause."""
        test = ContainerPauseTest(chaos_config)
        result = test.run()
        assert result.passed, f"Container pause test failed: {result.error_message}"
        assert result.recovery_time_seconds < 30, "Recovery took too long"

    def test_database_reconnection(self, chaos_config):
        """Test system handles database connection loss."""
        test = DatabaseConnectionTest(chaos_config)
        result = test.run()
        assert result.passed, f"Database test failed: {result.error_message}"

    def test_redis_unavailable(self, chaos_config):
        """Test system degrades gracefully without Redis."""
        test = RedisConnectionTest(chaos_config)
        result = test.run()
        assert result.passed, f"Redis test failed: {result.error_message}"


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    config = ChaosConfig()
    runner = ChaosTestRunner(config)

    # Run specific test or all tests
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        result = runner.run_test(test_name)
        if result:
            print(f"\n{test_name}: {'PASS' if result.passed else 'FAIL'}")
        else:
            print(f"Test not found: {test_name}")
    else:
        runner.run_all()
        runner.print_report()
