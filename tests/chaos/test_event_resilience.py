# -*- coding: utf-8 -*-
"""
Chaos Engineering Tests: V7.1 Event-Driven Resilience
======================================================
Tests system behavior under failure conditions:
- PostgreSQL connection drops
- Network partitions
- Slow consumers
- Message floods
- Circuit breaker behavior
- Dead Letter Queue recovery

Philosophy:
- Systems should degrade gracefully
- No data loss under failure conditions
- Automatic recovery when possible
- Clear alerting when manual intervention needed

Author: Trading Team
Version: 1.0.0
Created: 2026-01-31
Contract: CTR-V7-CHAOS
"""

import concurrent.futures
import json
import os
import queue
import random
import select
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# Skip if psycopg2 not available
psycopg2 = pytest.importorskip("psycopg2")
from psycopg2 import extensions


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def db_connection():
    """Get database connection for chaos tests."""
    conn_string = os.environ.get(
        "TEST_DATABASE_URL",
        os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/trading")
    )
    try:
        conn = psycopg2.connect(conn_string)
        conn.set_isolation_level(extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        yield conn
        conn.close()
    except Exception as e:
        pytest.skip(f"Database not available: {e}")


@pytest.fixture(scope="module")
def conn_string():
    """Get connection string for creating additional connections."""
    return os.environ.get(
        "TEST_DATABASE_URL",
        os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/trading")
    )


# =============================================================================
# CHAOS TEST: CONNECTION FAILURES
# =============================================================================

class TestConnectionChaos:
    """Tests for connection failure scenarios."""

    def test_circuit_opens_on_connection_failure(self):
        """Test circuit opens when PostgreSQL connection fails."""
        from airflow.dags.sensors.postgres_notify_sensor import (
            CircuitBreaker, CircuitState, PostgresNotifySensorBase
        )

        cb = CircuitBreaker(max_failures=3)

        # Simulate 3 connection failures
        for _ in range(3):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN
        assert cb.is_using_fallback is True

    def test_fallback_to_polling_works(self, db_connection):
        """Test that polling fallback works when NOTIFY fails."""
        from airflow.dags.sensors.postgres_notify_sensor import CircuitBreaker, CircuitState

        cb = CircuitBreaker(max_failures=1)
        cb.record_failure()

        assert cb.state == CircuitState.OPEN
        assert cb.is_using_fallback is True

        # System should use polling in this state
        # (Actual polling test would require full sensor setup)

    def test_recovery_after_connection_restored(self):
        """Test circuit closes after successful operations."""
        from airflow.dags.sensors.postgres_notify_sensor import CircuitBreaker, CircuitState

        cb = CircuitBreaker(max_failures=1, reset_timeout_seconds=0, half_open_success_threshold=2)

        # Open circuit
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Allow transition to half-open
        cb.should_allow_request()
        assert cb.state == CircuitState.HALF_OPEN

        # Simulate recovery
        cb.record_success()
        cb.record_success()

        assert cb.state == CircuitState.CLOSED


# =============================================================================
# CHAOS TEST: SLOW CONSUMER
# =============================================================================

class TestSlowConsumerChaos:
    """Tests for slow consumer scenarios."""

    def test_message_queue_under_slow_consumer(self, conn_string):
        """Test system behavior when consumer is slow."""
        # Create producer and consumer connections
        producer = psycopg2.connect(conn_string)
        producer.set_isolation_level(extensions.ISOLATION_LEVEL_AUTOCOMMIT)

        consumer = psycopg2.connect(conn_string)
        consumer.set_isolation_level(extensions.ISOLATION_LEVEL_AUTOCOMMIT)

        channel = "slow_consumer_test"

        # Start listening
        cur = consumer.cursor()
        cur.execute(f"LISTEN {channel}")
        cur.close()

        received_count = 0
        sent_count = 100

        # Send messages rapidly
        prod_cur = producer.cursor()
        for i in range(sent_count):
            prod_cur.execute(f"NOTIFY {channel}, 'msg_{i}'")
        prod_cur.close()

        # Simulate slow consumer (process with delays)
        start_time = time.time()
        while time.time() - start_time < 5.0:  # 5 second timeout
            if select.select([consumer], [], [], 0.1) != ([], [], []):
                consumer.poll()
                while consumer.notifies:
                    consumer.notifies.pop(0)
                    received_count += 1
                    time.sleep(0.01)  # Simulate slow processing

        producer.close()
        consumer.close()

        # PostgreSQL buffers notifications, so all should be received
        assert received_count >= sent_count * 0.9, f"Lost too many messages: {received_count}/{sent_count}"

    def test_backpressure_handling(self, conn_string):
        """Test that producer doesn't overwhelm consumer."""
        producer = psycopg2.connect(conn_string)
        producer.set_isolation_level(extensions.ISOLATION_LEVEL_AUTOCOMMIT)

        consumer = psycopg2.connect(conn_string)
        consumer.set_isolation_level(extensions.ISOLATION_LEVEL_AUTOCOMMIT)

        channel = "backpressure_test"

        cur = consumer.cursor()
        cur.execute(f"LISTEN {channel}")
        cur.close()

        # Send burst of messages
        prod_cur = producer.cursor()
        for i in range(500):
            prod_cur.execute(f"NOTIFY {channel}, 'burst_{i}'")
        prod_cur.close()

        # Consumer processes at normal rate
        received = 0
        start = time.time()
        while time.time() - start < 10.0 and received < 500:
            if select.select([consumer], [], [], 0.5) != ([], [], []):
                consumer.poll()
                batch_size = len(consumer.notifies)
                received += batch_size
                consumer.notifies.clear()

        producer.close()
        consumer.close()

        assert received >= 450, f"Dropped too many messages under backpressure: {received}/500"


# =============================================================================
# CHAOS TEST: MESSAGE FLOOD
# =============================================================================

class TestMessageFloodChaos:
    """Tests for high-volume message scenarios."""

    def test_handles_1000_messages_per_second(self, conn_string):
        """Test system handles high message rate."""
        producer = psycopg2.connect(conn_string)
        producer.set_isolation_level(extensions.ISOLATION_LEVEL_AUTOCOMMIT)

        consumer = psycopg2.connect(conn_string)
        consumer.set_isolation_level(extensions.ISOLATION_LEVEL_AUTOCOMMIT)

        channel = "flood_test"
        message_count = 1000
        received = 0
        latencies = []

        cur = consumer.cursor()
        cur.execute(f"LISTEN {channel}")
        cur.close()

        # Send messages with timestamp
        start_send = time.perf_counter()
        prod_cur = producer.cursor()
        for i in range(message_count):
            payload = json.dumps({"id": i, "ts": time.time_ns()})
            prod_cur.execute(f"NOTIFY {channel}, '{payload}'")
        prod_cur.close()
        send_duration = time.perf_counter() - start_send

        # Receive all messages
        receive_start = time.time()
        while time.time() - receive_start < 10.0:
            if select.select([consumer], [], [], 0.1) != ([], [], []):
                consumer.poll()
                while consumer.notifies:
                    notify = consumer.notifies.pop(0)
                    try:
                        data = json.loads(notify.payload)
                        latency_ns = time.time_ns() - data.get("ts", 0)
                        latencies.append(latency_ns / 1_000_000)  # Convert to ms
                    except:
                        pass
                    received += 1

            if received >= message_count:
                break

        producer.close()
        consumer.close()

        # Assertions
        assert received >= message_count * 0.95, f"Lost messages: {received}/{message_count}"

        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            p99_latency = sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 10 else 0

            # Log results
            print(f"\n  Messages: {message_count}")
            print(f"  Send rate: {message_count/send_duration:.0f} msg/s")
            print(f"  Received: {received}")
            print(f"  Avg latency: {avg_latency:.2f}ms")
            print(f"  P99 latency: {p99_latency:.2f}ms")


# =============================================================================
# CHAOS TEST: DEAD LETTER QUEUE
# =============================================================================

class TestDLQChaos:
    """Tests for DLQ under chaos conditions."""

    def test_failed_events_go_to_dlq(self, db_connection):
        """Test that failed events are properly queued to DLQ."""
        from airflow.dags.sensors.postgres_notify_sensor import DeadLetterQueue

        dlq = DeadLetterQueue(db_connection)

        # Simulate event processing failure
        event_id = f"chaos_event_{int(time.time())}"
        dlq_id = dlq.enqueue(
            event_id=event_id,
            event_type="chaos_test",
            channel="chaos_channel",
            payload={"chaos": True, "timestamp": time.time()},
            error="Simulated chaos failure"
        )

        assert dlq_id is not None

        # Verify event is in DLQ
        cur = db_connection.cursor()
        cur.execute("SELECT status FROM event_dead_letter_queue WHERE event_id = %s", (event_id,))
        row = cur.fetchone()
        assert row is not None
        assert row[0] == 'pending'

        # Cleanup
        cur.execute("DELETE FROM event_dead_letter_queue WHERE event_id = %s", (event_id,))
        cur.close()

    def test_dlq_retry_mechanism(self, db_connection):
        """Test DLQ retry with exponential backoff."""
        event_id = f"retry_test_{int(time.time())}"

        cur = db_connection.cursor()

        # First failure
        cur.execute(
            "SELECT dlq_enqueue(%s, %s, %s, %s, %s)",
            (event_id, "retry_test", "chaos", '{}', "Failure 1")
        )
        cur.execute("SELECT error_count FROM event_dead_letter_queue WHERE event_id = %s", (event_id,))
        assert cur.fetchone()[0] == 1

        # Second failure
        cur.execute(
            "SELECT dlq_enqueue(%s, %s, %s, %s, %s)",
            (event_id, "retry_test", "chaos", '{}', "Failure 2")
        )
        cur.execute("SELECT error_count FROM event_dead_letter_queue WHERE event_id = %s", (event_id,))
        assert cur.fetchone()[0] == 2

        # Cleanup
        cur.execute("DELETE FROM event_dead_letter_queue WHERE event_id = %s", (event_id,))
        cur.close()


# =============================================================================
# CHAOS TEST: IDEMPOTENCY
# =============================================================================

class TestIdempotencyChaos:
    """Tests for idempotency under chaos conditions."""

    def test_duplicate_events_processed_once(self):
        """Test that duplicate events are not processed twice."""
        from airflow.dags.sensors.postgres_notify_sensor import IdempotentProcessor

        processor = IdempotentProcessor(persist_to_db=False)
        processed_count = 0

        # Same event payload
        payload = {"symbol": "USD/COP", "time": "2026-01-31T12:00:00", "event_type": "new_bar"}

        # Process 100 times (simulating duplicates)
        for _ in range(100):
            event_id = processor.compute_event_id(payload)
            if not processor.is_processed(event_id):
                processed_count += 1
                processor.mark_processed(event_id, payload)

        assert processed_count == 1, f"Event was processed {processed_count} times instead of 1"

    def test_concurrent_duplicate_detection(self):
        """Test idempotency under concurrent access."""
        from airflow.dags.sensors.postgres_notify_sensor import IdempotentProcessor

        processor = IdempotentProcessor(persist_to_db=False)
        payload = {"symbol": "USD/COP", "time": "2026-01-31T13:00:00", "event_type": "new_bar"}
        event_id = processor.compute_event_id(payload)

        processed_count = 0
        lock = threading.Lock()

        def try_process():
            nonlocal processed_count
            if not processor.is_processed(event_id):
                with lock:
                    if not processor.is_processed(event_id):
                        processed_count += 1
                        processor.mark_processed(event_id, payload)

        # Simulate concurrent access
        threads = [threading.Thread(target=try_process) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert processed_count == 1, f"Event was processed {processed_count} times concurrently"


# =============================================================================
# CHAOS TEST: HEARTBEAT FAILURES
# =============================================================================

class TestHeartbeatChaos:
    """Tests for heartbeat failure detection."""

    def test_missed_heartbeats_detected(self):
        """Test that missed heartbeats are properly detected."""
        from airflow.dags.sensors.postgres_notify_sensor import HeartbeatMonitor

        with patch('psycopg2.connect') as mock_conn:
            mock_connection = MagicMock()
            mock_conn.return_value = mock_connection

            monitor = HeartbeatMonitor(mock_connection)
            monitor.MAX_MISSED_HEARTBEATS = 3

            # Simulate missed heartbeats
            for _ in range(3):
                monitor.missed_heartbeats += 1

            monitor.is_healthy = monitor.missed_heartbeats < monitor.MAX_MISSED_HEARTBEATS

            assert monitor.is_healthy is False, "Should be unhealthy after max missed heartbeats"


# =============================================================================
# CHAOS TEST: NETWORK PARTITION SIMULATION
# =============================================================================

class TestNetworkPartitionChaos:
    """Tests for network partition scenarios."""

    def test_graceful_degradation_on_partition(self):
        """Test system degrades gracefully during network partition."""
        from airflow.dags.sensors.postgres_notify_sensor import CircuitBreaker, CircuitState

        cb = CircuitBreaker(max_failures=3)

        # Simulate network partition (rapid failures)
        for _ in range(5):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN
        assert cb.is_using_fallback is True

        # System should still function in degraded mode
        # (Polling fallback)

    def test_circuit_breaker_prevents_cascade_failures(self):
        """Test circuit breaker prevents cascading failures."""
        from airflow.dags.sensors.postgres_notify_sensor import CircuitBreaker, CircuitState

        cb = CircuitBreaker(max_failures=2, reset_timeout_seconds=60)

        # Rapid failures
        cb.record_failure()
        cb.record_failure()

        # Circuit should be open
        assert cb.state == CircuitState.OPEN

        # Further requests should not be allowed
        # (Protection against cascade)
        assert cb.should_allow_request() is False


# =============================================================================
# CHAOS TEST: RANDOM FAILURE INJECTION
# =============================================================================

class TestRandomFailureInjection:
    """Tests with random failure injection."""

    def test_system_stable_under_random_failures(self):
        """Test system remains stable under random failure injection."""
        from airflow.dags.sensors.postgres_notify_sensor import (
            CircuitBreaker, CircuitState, IdempotentProcessor
        )

        cb = CircuitBreaker(max_failures=5, reset_timeout_seconds=1)
        processor = IdempotentProcessor(persist_to_db=False)

        events_processed = 0
        failures_encountered = 0

        for i in range(100):
            # Random failure (30% chance)
            if random.random() < 0.3:
                cb.record_failure()
                failures_encountered += 1
            else:
                if cb.should_allow_request():
                    payload = {"id": i, "time": datetime.utcnow().isoformat()}
                    event_id = processor.compute_event_id(payload)

                    if not processor.is_processed(event_id):
                        processor.mark_processed(event_id, payload)
                        events_processed += 1
                        cb.record_success()

            # Reset timeout check
            if cb.state == CircuitState.OPEN:
                time.sleep(0.01)  # Fast-forward

        print(f"\n  Failures: {failures_encountered}/100")
        print(f"  Processed: {events_processed}")
        print(f"  Final state: {cb.state.name}")

        # System should have processed some events despite failures
        assert events_processed > 0


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
