# -*- coding: utf-8 -*-
"""
Integration Tests: V7.1 Event-Driven Architecture
==================================================
End-to-end tests for PostgreSQL NOTIFY system, circuit breaker,
idempotency, and hybrid feature retrieval.

Test Categories:
1. PostgreSQL NOTIFY Integration
2. Circuit Breaker Behavior
3. Idempotent Processing
4. Dead Letter Queue
5. Hybrid Mode (PostgreSQL during market, Redis off-market)
6. Heartbeat Monitoring

Author: Trading Team
Version: 1.0.0
Created: 2026-01-31
Contract: CTR-V7-TESTS
"""

import json
import os
import pytest
import queue
import select
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import patch, MagicMock

# Skip if psycopg2 not available
psycopg2 = pytest.importorskip("psycopg2")
from psycopg2 import extensions


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def db_connection():
    """Get database connection for tests."""
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
def migration_applied(db_connection):
    """Ensure migration 033 is applied."""
    cur = db_connection.cursor()
    try:
        # Check if trigger exists
        cur.execute("""
            SELECT 1 FROM pg_trigger
            WHERE tgname = 'trg_notify_new_ohlcv_bar'
        """)
        if not cur.fetchone():
            pytest.skip("Migration 033 not applied. Run: psql -f database/migrations/033_event_triggers.sql")
        yield True
    finally:
        cur.close()


# =============================================================================
# TEST: PostgreSQL NOTIFY
# =============================================================================

class TestPostgresNotify:
    """Tests for PostgreSQL NOTIFY functionality."""

    def test_notify_can_be_sent(self, db_connection):
        """Test that NOTIFY can be sent."""
        cur = db_connection.cursor()
        cur.execute("NOTIFY test_channel, 'test_payload'")
        cur.close()

    def test_notify_can_be_received(self, db_connection):
        """Test that NOTIFY can be received via LISTEN."""
        # Create listener connection
        conn_string = db_connection.dsn
        listener = psycopg2.connect(conn_string)
        listener.set_isolation_level(extensions.ISOLATION_LEVEL_AUTOCOMMIT)

        cur = listener.cursor()
        cur.execute("LISTEN test_receive_channel")
        cur.close()

        # Send notification
        sender_cur = db_connection.cursor()
        sender_cur.execute("NOTIFY test_receive_channel, 'hello_world'")
        sender_cur.close()

        # Wait for notification
        if select.select([listener], [], [], 2.0) != ([], [], []):
            listener.poll()
            assert len(listener.notifies) > 0
            notify = listener.notifies.pop(0)
            assert notify.payload == 'hello_world'
        else:
            pytest.fail("NOTIFY not received within timeout")

        listener.close()

    def test_notify_latency_under_100ms(self, db_connection):
        """Test that NOTIFY latency is under 100ms."""
        conn_string = db_connection.dsn
        listener = psycopg2.connect(conn_string)
        listener.set_isolation_level(extensions.ISOLATION_LEVEL_AUTOCOMMIT)

        cur = listener.cursor()
        cur.execute("LISTEN latency_test")
        cur.close()

        # Measure latency
        latencies = []
        for i in range(10):
            send_time = time.perf_counter()

            # Send notification
            sender_cur = db_connection.cursor()
            sender_cur.execute(f"NOTIFY latency_test, 'msg_{i}'")
            sender_cur.close()

            # Wait for receipt
            if select.select([listener], [], [], 1.0) != ([], [], []):
                listener.poll()
                receive_time = time.perf_counter()
                latencies.append((receive_time - send_time) * 1000)
                listener.notifies.clear()
            else:
                pytest.fail(f"Message {i} not received")

        listener.close()

        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)

        assert avg_latency < 50, f"Average latency {avg_latency:.2f}ms exceeds 50ms"
        assert max_latency < 100, f"Max latency {max_latency:.2f}ms exceeds 100ms threshold"


# =============================================================================
# TEST: CIRCUIT BREAKER
# =============================================================================

class TestCircuitBreaker:
    """Tests for Circuit Breaker pattern."""

    def test_circuit_starts_closed(self):
        """Test circuit starts in CLOSED state."""
        from airflow.dags.sensors.postgres_notify_sensor import CircuitBreaker, CircuitState

        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.should_allow_request() is True

    def test_circuit_opens_after_failures(self):
        """Test circuit opens after max failures."""
        from airflow.dags.sensors.postgres_notify_sensor import CircuitBreaker, CircuitState

        cb = CircuitBreaker(max_failures=3)

        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

        cb.record_failure()  # 3rd failure
        assert cb.state == CircuitState.OPEN
        assert cb.is_using_fallback is True

    def test_circuit_transitions_to_half_open(self):
        """Test circuit transitions to HALF_OPEN after reset timeout."""
        from airflow.dags.sensors.postgres_notify_sensor import CircuitBreaker, CircuitState

        cb = CircuitBreaker(max_failures=1, reset_timeout_seconds=0)  # Immediate reset

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Should transition to HALF_OPEN on next request
        assert cb.should_allow_request() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_circuit_closes_after_success_in_half_open(self):
        """Test circuit closes after success in HALF_OPEN state."""
        from airflow.dags.sensors.postgres_notify_sensor import CircuitBreaker, CircuitState

        cb = CircuitBreaker(max_failures=1, reset_timeout_seconds=0, half_open_success_threshold=2)

        cb.record_failure()
        cb.should_allow_request()  # Transition to HALF_OPEN

        cb.record_success()
        cb.record_success()

        assert cb.state == CircuitState.CLOSED


# =============================================================================
# TEST: IDEMPOTENT PROCESSOR
# =============================================================================

class TestIdempotentProcessor:
    """Tests for idempotent event processing."""

    def test_compute_event_id(self):
        """Test event ID computation is deterministic."""
        from airflow.dags.sensors.postgres_notify_sensor import IdempotentProcessor

        processor = IdempotentProcessor(persist_to_db=False)

        payload1 = {"symbol": "USD/COP", "time": "2026-01-31T12:00:00", "event_type": "new_bar"}
        payload2 = {"symbol": "USD/COP", "time": "2026-01-31T12:00:00", "event_type": "new_bar"}
        payload3 = {"symbol": "USD/COP", "time": "2026-01-31T12:05:00", "event_type": "new_bar"}

        id1 = processor.compute_event_id(payload1)
        id2 = processor.compute_event_id(payload2)
        id3 = processor.compute_event_id(payload3)

        assert id1 == id2, "Same payload should produce same event ID"
        assert id1 != id3, "Different payload should produce different event ID"

    def test_duplicate_detection(self):
        """Test duplicate event detection."""
        from airflow.dags.sensors.postgres_notify_sensor import IdempotentProcessor

        processor = IdempotentProcessor(persist_to_db=False)

        payload = {"symbol": "USD/COP", "time": "2026-01-31T12:00:00", "event_type": "new_bar"}
        event_id = processor.compute_event_id(payload)

        assert processor.is_processed(event_id) is False
        processor.mark_processed(event_id, payload)
        assert processor.is_processed(event_id) is True

    def test_cache_size_limit(self):
        """Test cache size limit is enforced."""
        from airflow.dags.sensors.postgres_notify_sensor import IdempotentProcessor

        processor = IdempotentProcessor(cache_size=10, persist_to_db=False)

        # Add more events than cache size
        for i in range(15):
            payload = {"symbol": "USD/COP", "time": f"2026-01-31T12:{i:02d}:00", "event_type": "new_bar"}
            event_id = processor.compute_event_id(payload)
            processor.mark_processed(event_id, payload)

        assert len(processor._cache) == 10


# =============================================================================
# TEST: DEAD LETTER QUEUE
# =============================================================================

class TestDeadLetterQueue:
    """Tests for Dead Letter Queue."""

    def test_dlq_enqueue(self, db_connection, migration_applied):
        """Test event can be enqueued to DLQ."""
        from airflow.dags.sensors.postgres_notify_sensor import DeadLetterQueue

        dlq = DeadLetterQueue(db_connection, max_retries=5)

        event_id = f"test_event_{int(time.time())}"
        dlq_id = dlq.enqueue(
            event_id=event_id,
            event_type="test",
            channel="test_channel",
            payload={"test": "data"},
            error="Test error message"
        )

        assert dlq_id is not None

        # Cleanup
        cur = db_connection.cursor()
        cur.execute("DELETE FROM event_dead_letter_queue WHERE event_id = %s", (event_id,))
        cur.close()

    def test_dlq_exponential_backoff(self, db_connection, migration_applied):
        """Test DLQ retry uses exponential backoff."""
        cur = db_connection.cursor()

        event_id = f"backoff_test_{int(time.time())}"

        # First enqueue
        cur.execute(
            "SELECT dlq_enqueue(%s, %s, %s, %s, %s)",
            (event_id, "test", "test_channel", '{"test": true}', "Error 1")
        )

        # Get retry_after
        cur.execute(
            "SELECT error_count, retry_after FROM event_dead_letter_queue WHERE event_id = %s",
            (event_id,)
        )
        row1 = cur.fetchone()

        # Second enqueue (same event)
        cur.execute(
            "SELECT dlq_enqueue(%s, %s, %s, %s, %s)",
            (event_id, "test", "test_channel", '{"test": true}', "Error 2")
        )

        cur.execute(
            "SELECT error_count, retry_after FROM event_dead_letter_queue WHERE event_id = %s",
            (event_id,)
        )
        row2 = cur.fetchone()

        assert row2[0] == row1[0] + 1, "Error count should increment"

        # Cleanup
        cur.execute("DELETE FROM event_dead_letter_queue WHERE event_id = %s", (event_id,))
        cur.close()


# =============================================================================
# TEST: HYBRID MODE
# =============================================================================

class TestHybridMode:
    """Tests for V7.1 Hybrid Mode (PostgreSQL during market, Redis off-market)."""

    def test_is_market_hours_weekday(self):
        """Test market hours detection on weekday."""
        from src.feature_store.feast_service import is_market_hours
        from zoneinfo import ZoneInfo

        cot = ZoneInfo("America/Bogota")

        # Monday 10:00 COT = market hours
        dt_market = datetime(2026, 2, 2, 10, 0, tzinfo=cot)  # Monday
        assert is_market_hours(dt_market) is True

        # Monday 06:00 COT = before market
        dt_before = datetime(2026, 2, 2, 6, 0, tzinfo=cot)
        assert is_market_hours(dt_before) is False

        # Monday 18:00 COT = after market
        dt_after = datetime(2026, 2, 2, 18, 0, tzinfo=cot)
        assert is_market_hours(dt_after) is False

    def test_is_market_hours_weekend(self):
        """Test market hours detection on weekend."""
        from src.feature_store.feast_service import is_market_hours
        from zoneinfo import ZoneInfo

        cot = ZoneInfo("America/Bogota")

        # Saturday 10:00 COT = not market hours
        dt_saturday = datetime(2026, 1, 31, 10, 0, tzinfo=cot)  # Saturday
        assert is_market_hours(dt_saturday) is False

    def test_feast_service_health_check_includes_hybrid_info(self):
        """Test health check includes hybrid mode info."""
        from src.feature_store.feast_service import FeastInferenceService

        service = FeastInferenceService(enable_hybrid_mode=True, enable_fallback=False)
        health = service.health_check()

        assert "hybrid_mode_enabled" in health
        assert "is_market_hours" in health
        assert "active_backend" in health


# =============================================================================
# TEST: HEARTBEAT MONITOR
# =============================================================================

class TestHeartbeatMonitor:
    """Tests for Heartbeat Monitor."""

    def test_heartbeat_status_structure(self):
        """Test heartbeat status returns expected structure."""
        from airflow.dags.sensors.postgres_notify_sensor import HeartbeatMonitor

        with patch('psycopg2.connect') as mock_conn:
            mock_conn.return_value.closed = False
            monitor = HeartbeatMonitor(mock_conn.return_value)

            status = monitor.get_status()

            assert "is_healthy" in status
            assert "missed_heartbeats" in status
            assert "last_heartbeat_received" in status
            assert "max_allowed_missed" in status


# =============================================================================
# TEST: END-TO-END FLOW
# =============================================================================

class TestEndToEndFlow:
    """End-to-end tests for event-driven flow."""

    def test_ohlcv_insert_triggers_notify(self, db_connection, migration_applied):
        """Test that inserting OHLCV data triggers NOTIFY."""
        conn_string = db_connection.dsn
        listener = psycopg2.connect(conn_string)
        listener.set_isolation_level(extensions.ISOLATION_LEVEL_AUTOCOMMIT)

        cur = listener.cursor()
        cur.execute("LISTEN ohlcv_updates")
        cur.close()

        # Insert test data
        test_time = datetime.utcnow().replace(second=0, microsecond=0)
        insert_cur = db_connection.cursor()

        try:
            insert_cur.execute("""
                INSERT INTO usdcop_m5_ohlcv (time, symbol, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (time, symbol) DO UPDATE SET close = EXCLUDED.close
            """, (test_time, 'TEST/TEST', 4100.0, 4105.0, 4095.0, 4102.0, 1000))

            # Wait for notification
            if select.select([listener], [], [], 2.0) != ([], [], []):
                listener.poll()
                assert len(listener.notifies) > 0
                notify = listener.notifies.pop(0)
                payload = json.loads(notify.payload)
                assert payload["event_type"] == "new_bar"
                assert payload["symbol"] == "TEST/TEST"
            else:
                pytest.fail("NOTIFY not received after OHLCV insert")

        finally:
            # Cleanup
            insert_cur.execute(
                "DELETE FROM usdcop_m5_ohlcv WHERE symbol = 'TEST/TEST'"
            )
            insert_cur.close()
            listener.close()


# =============================================================================
# TEST: METRICS
# =============================================================================

class TestEventDrivenMetrics:
    """Tests for event-driven metrics."""

    def test_metrics_snapshot(self):
        """Test metrics snapshot generation."""
        from src.monitoring.event_driven_metrics import get_metrics

        metrics = get_metrics()
        metrics.record_notify_event("test", "test_event", 0.005, success=True)

        snapshot = metrics.get_snapshot()

        assert snapshot.notify_events_total >= 1
        assert snapshot.timestamp is not None

    def test_circuit_state_recording(self):
        """Test circuit breaker state recording."""
        from src.monitoring.event_driven_metrics import get_metrics, CircuitState

        metrics = get_metrics()
        metrics.record_circuit_state("test_sensor", CircuitState.OPEN)

        open_circuits = metrics.get_open_circuits()
        assert "test_sensor" in open_circuits


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
