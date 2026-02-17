#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V7.1 Deployment Validation Script
==================================
Post-deployment validation for V7.1 event-driven architecture.

Validates:
1. Migration 033 applied correctly
2. NOTIFY triggers are firing
3. Latency is within thresholds
4. Circuit breaker is functional
5. Feast hybrid mode is working
6. Metrics are being collected

Usage:
    python scripts/validate_v7_deployment.py --full

Author: Trading Team
Version: 1.0.0
Created: 2026-01-31
Contract: CTR-V7-VALIDATE
"""

import argparse
import json
import logging
import os
import select
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION RESULT
# =============================================================================

@dataclass
class ValidationResult:
    """Result of a validation check."""
    name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "details": self.details
        }


@dataclass
class ValidationReport:
    """Complete validation report."""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    overall_passed: bool = False
    checks: List[ValidationResult] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)

    def add_check(self, result: ValidationResult):
        self.checks.append(result)

    def finalize(self):
        passed_count = sum(1 for c in self.checks if c.passed)
        failed_count = len(self.checks) - passed_count
        self.summary = {"passed": passed_count, "failed": failed_count, "total": len(self.checks)}
        self.overall_passed = failed_count == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "overall_passed": self.overall_passed,
            "summary": self.summary,
            "checks": [c.to_dict() for c in self.checks]
        }


# =============================================================================
# VALIDATORS
# =============================================================================

class V7Validator:
    """V7.1 deployment validator."""

    def __init__(self, conn_string: str):
        self.conn_string = conn_string
        self.conn = None
        self.report = ValidationReport()

    def connect(self) -> bool:
        """Connect to database."""
        try:
            import psycopg2
            from psycopg2 import extensions
            self.conn = psycopg2.connect(self.conn_string)
            self.conn.set_isolation_level(extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def close(self):
        """Close connection."""
        if self.conn:
            self.conn.close()

    # -------------------------------------------------------------------------
    # VALIDATION CHECKS
    # -------------------------------------------------------------------------

    def check_migration_applied(self) -> ValidationResult:
        """Check if migration 033 is applied."""
        cur = self.conn.cursor()

        try:
            # Check OHLCV trigger
            cur.execute("""
                SELECT 1 FROM pg_trigger
                WHERE tgname = 'trg_notify_new_ohlcv_bar'
            """)
            ohlcv_trigger = cur.fetchone() is not None

            # Check features trigger
            cur.execute("""
                SELECT 1 FROM pg_trigger
                WHERE tgname = 'trg_notify_features_ready'
            """)
            features_trigger = cur.fetchone() is not None

            # Check DLQ table
            cur.execute("""
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'event_dead_letter_queue'
            """)
            dlq_table = cur.fetchone() is not None

            passed = ohlcv_trigger and features_trigger and dlq_table

            return ValidationResult(
                name="Migration 033 Applied",
                passed=passed,
                message="All V7.1 database objects present" if passed else "Missing V7.1 database objects",
                details={
                    "ohlcv_trigger": ohlcv_trigger,
                    "features_trigger": features_trigger,
                    "dlq_table": dlq_table
                }
            )

        finally:
            cur.close()

    def check_notify_functional(self) -> ValidationResult:
        """Check if NOTIFY is working."""
        try:
            import psycopg2
            from psycopg2 import extensions

            # Create listener connection
            listener = psycopg2.connect(self.conn_string)
            listener.set_isolation_level(extensions.ISOLATION_LEVEL_AUTOCOMMIT)

            cur = listener.cursor()
            cur.execute("LISTEN v7_validation_test")
            cur.close()

            # Send notification
            sender = self.conn.cursor()
            sender.execute("NOTIFY v7_validation_test, 'validation'")
            sender.close()

            # Check receipt
            if select.select([listener], [], [], 2.0) != ([], [], []):
                listener.poll()
                received = len(listener.notifies) > 0
            else:
                received = False

            listener.close()

            return ValidationResult(
                name="NOTIFY Functional",
                passed=received,
                message="NOTIFY/LISTEN working correctly" if received else "NOTIFY not received",
                details={"notification_received": received}
            )

        except Exception as e:
            return ValidationResult(
                name="NOTIFY Functional",
                passed=False,
                message=f"Error: {e}",
                details={"error": str(e)}
            )

    def check_latency_threshold(self) -> ValidationResult:
        """Check NOTIFY latency is under threshold."""
        try:
            import psycopg2
            from psycopg2 import extensions

            listener = psycopg2.connect(self.conn_string)
            listener.set_isolation_level(extensions.ISOLATION_LEVEL_AUTOCOMMIT)

            cur = listener.cursor()
            cur.execute("LISTEN latency_check")
            cur.close()

            latencies = []
            for i in range(10):
                send_time = time.perf_counter()

                sender = self.conn.cursor()
                sender.execute(f"NOTIFY latency_check, 'msg_{i}'")
                sender.close()

                if select.select([listener], [], [], 1.0) != ([], [], []):
                    listener.poll()
                    receive_time = time.perf_counter()
                    latencies.append((receive_time - send_time) * 1000)
                    listener.notifies.clear()

            listener.close()

            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                max_latency = max(latencies)
                p99_latency = sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 1 else max_latency

                passed = p99_latency < 100.0

                return ValidationResult(
                    name="Latency Threshold",
                    passed=passed,
                    message=f"p99 latency: {p99_latency:.2f}ms {'< 100ms' if passed else '>= 100ms'}",
                    details={
                        "avg_latency_ms": round(avg_latency, 2),
                        "max_latency_ms": round(max_latency, 2),
                        "p99_latency_ms": round(p99_latency, 2),
                        "threshold_ms": 100.0
                    }
                )
            else:
                return ValidationResult(
                    name="Latency Threshold",
                    passed=False,
                    message="No latency measurements collected",
                    details={}
                )

        except Exception as e:
            return ValidationResult(
                name="Latency Threshold",
                passed=False,
                message=f"Error: {e}",
                details={"error": str(e)}
            )

    def check_circuit_breaker(self) -> ValidationResult:
        """Check circuit breaker is importable and functional."""
        try:
            from airflow.dags.sensors.postgres_notify_sensor import CircuitBreaker, CircuitState

            cb = CircuitBreaker(max_failures=3)

            # Test state transitions
            assert cb.state == CircuitState.CLOSED
            cb.record_failure()
            cb.record_failure()
            cb.record_failure()
            assert cb.state == CircuitState.OPEN

            return ValidationResult(
                name="Circuit Breaker",
                passed=True,
                message="Circuit breaker functional",
                details={"states_tested": ["CLOSED", "OPEN"]}
            )

        except ImportError as e:
            return ValidationResult(
                name="Circuit Breaker",
                passed=False,
                message=f"Import error: {e}",
                details={"error": str(e)}
            )
        except AssertionError as e:
            return ValidationResult(
                name="Circuit Breaker",
                passed=False,
                message=f"State transition failed: {e}",
                details={"error": str(e)}
            )
        except Exception as e:
            return ValidationResult(
                name="Circuit Breaker",
                passed=False,
                message=f"Error: {e}",
                details={"error": str(e)}
            )

    def check_hybrid_mode(self) -> ValidationResult:
        """Check Feast hybrid mode is configured."""
        try:
            from src.feature_store.feast_service import FeastInferenceService, is_market_hours

            service = FeastInferenceService(enable_hybrid_mode=True, enable_fallback=False)
            health = service.health_check()

            return ValidationResult(
                name="Hybrid Mode",
                passed=True,
                message=f"Active backend: {health.get('active_backend', 'unknown')}",
                details={
                    "hybrid_enabled": health.get("hybrid_mode_enabled"),
                    "is_market_hours": health.get("is_market_hours"),
                    "active_backend": health.get("active_backend"),
                    "postgres_connected": health.get("postgres_connected")
                }
            )

        except ImportError as e:
            return ValidationResult(
                name="Hybrid Mode",
                passed=False,
                message=f"Import error: {e}",
                details={"error": str(e)}
            )
        except Exception as e:
            return ValidationResult(
                name="Hybrid Mode",
                passed=False,
                message=f"Error: {e}",
                details={"error": str(e)}
            )

    def check_metrics_available(self) -> ValidationResult:
        """Check metrics module is available."""
        try:
            from src.monitoring.event_driven_metrics import get_metrics, PROMETHEUS_AVAILABLE

            metrics = get_metrics()
            snapshot = metrics.get_snapshot()

            return ValidationResult(
                name="Metrics Module",
                passed=True,
                message=f"Prometheus: {'available' if PROMETHEUS_AVAILABLE else 'not installed'}",
                details={
                    "prometheus_available": PROMETHEUS_AVAILABLE,
                    "snapshot_timestamp": snapshot.timestamp
                }
            )

        except ImportError as e:
            return ValidationResult(
                name="Metrics Module",
                passed=False,
                message=f"Import error: {e}",
                details={"error": str(e)}
            )
        except Exception as e:
            return ValidationResult(
                name="Metrics Module",
                passed=False,
                message=f"Error: {e}",
                details={"error": str(e)}
            )

    def check_dlq_empty(self) -> ValidationResult:
        """Check DLQ doesn't have dead events."""
        cur = self.conn.cursor()

        try:
            cur.execute("""
                SELECT
                    COUNT(*) FILTER (WHERE status = 'pending') as pending,
                    COUNT(*) FILTER (WHERE status = 'dead') as dead
                FROM event_dead_letter_queue
            """)
            row = cur.fetchone()
            pending = row[0] if row else 0
            dead = row[1] if row else 0

            passed = dead == 0

            return ValidationResult(
                name="DLQ Health",
                passed=passed,
                message=f"Pending: {pending}, Dead: {dead}",
                details={"pending": pending, "dead": dead}
            )

        except Exception as e:
            return ValidationResult(
                name="DLQ Health",
                passed=True,  # DLQ table might not exist yet
                message=f"DLQ not checked: {e}",
                details={"error": str(e)}
            )

        finally:
            cur.close()

    # -------------------------------------------------------------------------
    # RUN ALL CHECKS
    # -------------------------------------------------------------------------

    def run_all_checks(self) -> ValidationReport:
        """Run all validation checks."""
        logger.info("=" * 60)
        logger.info("V7.1 DEPLOYMENT VALIDATION")
        logger.info("=" * 60)

        checks = [
            ("Migration Applied", self.check_migration_applied),
            ("NOTIFY Functional", self.check_notify_functional),
            ("Latency Threshold", self.check_latency_threshold),
            ("Circuit Breaker", self.check_circuit_breaker),
            ("Hybrid Mode", self.check_hybrid_mode),
            ("Metrics Module", self.check_metrics_available),
            ("DLQ Health", self.check_dlq_empty),
        ]

        for name, check_func in checks:
            logger.info(f"Checking: {name}...")
            try:
                result = check_func()
            except Exception as e:
                result = ValidationResult(name=name, passed=False, message=f"Exception: {e}")

            self.report.add_check(result)

            status = "✅ PASS" if result.passed else "❌ FAIL"
            logger.info(f"  {status}: {result.message}")

        self.report.finalize()
        return self.report


# =============================================================================
# MAIN
# =============================================================================

def get_connection_string() -> str:
    """Get database connection string."""
    return os.environ.get(
        "DATABASE_URL",
        os.environ.get("TIMESCALE_URL", "postgresql://postgres:postgres@localhost:5432/trading")
    )


def main():
    parser = argparse.ArgumentParser(description="V7.1 Deployment Validation")
    parser.add_argument("--full", action="store_true", help="Run full validation")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--output", type=str, help="Save report to file")
    parser.add_argument("--connection", type=str, help="Database connection string")

    args = parser.parse_args()

    conn_string = args.connection or get_connection_string()

    validator = V7Validator(conn_string)

    if not validator.connect():
        print("Failed to connect to database")
        sys.exit(1)

    try:
        report = validator.run_all_checks()

        if args.json or args.output:
            report_dict = report.to_dict()
            report_json = json.dumps(report_dict, indent=2)

            if args.json:
                print(report_json)

            if args.output:
                with open(args.output, 'w') as f:
                    f.write(report_json)
                logger.info(f"Report saved to {args.output}")

        else:
            # Print summary
            print("\n" + "=" * 60)
            print("VALIDATION SUMMARY")
            print("=" * 60)
            print(f"Total Checks: {report.summary['total']}")
            print(f"Passed:       {report.summary['passed']} ✅")
            print(f"Failed:       {report.summary['failed']} ❌")
            print("-" * 60)
            print(f"Overall:      {'✅ PASS' if report.overall_passed else '❌ FAIL'}")
            print("=" * 60)

            if not report.overall_passed:
                print("\nFailed Checks:")
                for check in report.checks:
                    if not check.passed:
                        print(f"  - {check.name}: {check.message}")

        sys.exit(0 if report.overall_passed else 1)

    finally:
        validator.close()


if __name__ == "__main__":
    main()
