# -*- coding: utf-8 -*-
"""
PostgreSQL NOTIFY Sensors with Circuit Breaker - V7.1 Event-Driven Architecture
================================================================================
Replaces polling-based sensors with LISTEN/NOTIFY for near real-time response.

Features:
- PostgreSQL LISTEN for instant event notification
- Circuit Breaker pattern with automatic fallback to polling
- Idempotent event processing via event_id hash
- Dead Letter Queue integration for failed events
- Heartbeat monitoring for system health

Channels:
- ohlcv_updates: Triggered on new OHLCV bar insert
- feature_updates: Triggered when features are ready
- heartbeat: System health check

Author: Trading Team
Version: 2.0.0 (V7.1 Event-Driven)
Created: 2026-01-31
Contract: CTR-V7-SENSORS
"""

import hashlib
import json
import logging
import select
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from psycopg2 import extensions
from airflow.sensors.base import BaseSensorOperator
from airflow.utils.decorators import apply_defaults

logger = logging.getLogger(__name__)


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "CLOSED"      # Normal operation, NOTIFY active
    OPEN = "OPEN"          # Fallback to polling
    HALF_OPEN = "HALF_OPEN"  # Testing if NOTIFY recovered


@dataclass
class CircuitBreaker:
    """
    Circuit Breaker for PostgreSQL NOTIFY.

    When NOTIFY fails repeatedly, circuit opens and falls back to polling.
    After reset_timeout, attempts to use NOTIFY again (half-open state).
    """
    max_failures: int = 3
    reset_timeout_seconds: int = 300
    half_open_success_threshold: int = 2

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    opened_at: Optional[datetime] = None

    def record_success(self) -> None:
        """Record a successful operation."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_success_threshold:
                self._close()
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()

        if self.state == CircuitState.HALF_OPEN:
            self._open()
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.max_failures:
                self._open()

    def should_allow_request(self) -> bool:
        """Check if request should be allowed through circuit."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._half_open()
                return True
            return False

        # HALF_OPEN: allow limited requests
        return True

    def _open(self) -> None:
        """Open the circuit (switch to fallback)."""
        self.state = CircuitState.OPEN
        self.opened_at = datetime.utcnow()
        self.success_count = 0
        logger.warning(
            f"Circuit OPENED after {self.failure_count} failures. "
            f"Falling back to polling for {self.reset_timeout_seconds}s"
        )

    def _close(self) -> None:
        """Close the circuit (resume NOTIFY)."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.opened_at = None
        logger.info("Circuit CLOSED. NOTIFY resumed.")

    def _half_open(self) -> None:
        """Transition to half-open state."""
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        logger.info("Circuit HALF-OPEN. Testing NOTIFY...")

    def _should_attempt_reset(self) -> bool:
        """Check if enough time passed to attempt reset."""
        if not self.opened_at:
            return True
        elapsed = (datetime.utcnow() - self.opened_at).total_seconds()
        return elapsed >= self.reset_timeout_seconds

    @property
    def is_using_fallback(self) -> bool:
        """Check if currently using fallback (polling)."""
        return self.state == CircuitState.OPEN


# =============================================================================
# IDEMPOTENT PROCESSOR
# =============================================================================

@dataclass
class ProcessedEvent:
    """Tracks a processed event for idempotency."""
    event_id: str
    processed_at: datetime
    payload_hash: str


class IdempotentProcessor:
    """
    Ensures exactly-once event processing using hash-based deduplication.

    Events are identified by event_id (MD5 hash of key fields).
    Processed events are cached and can optionally persist to DB.
    """

    def __init__(self, cache_size: int = 10000, persist_to_db: bool = True):
        self.cache_size = cache_size
        self.persist_to_db = persist_to_db
        self._cache: Dict[str, ProcessedEvent] = {}
        self._db_conn = None

    @staticmethod
    def compute_event_id(payload: Dict[str, Any]) -> str:
        """Compute unique event ID from payload."""
        key_fields = ["symbol", "time", "event_type"]
        key_data = "|".join(str(payload.get(k, "")) for k in key_fields)
        return hashlib.md5(key_data.encode()).hexdigest()

    @staticmethod
    def compute_payload_hash(payload: Dict[str, Any]) -> str:
        """Compute hash of full payload for verification."""
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def is_processed(self, event_id: str) -> bool:
        """Check if event was already processed."""
        # Check cache first
        if event_id in self._cache:
            return True

        # Check DB if persisting
        if self.persist_to_db and self._db_conn:
            try:
                cur = self._db_conn.cursor()
                cur.execute("SELECT 1 FROM event_processed_log WHERE event_id = %s", (event_id,))
                result = cur.fetchone()
                cur.close()
                if result:
                    return True
            except Exception as e:
                logger.warning(f"DB check failed: {e}")

        return False

    def mark_processed(self, event_id: str, payload: Dict[str, Any]) -> None:
        """Mark event as processed."""
        event = ProcessedEvent(
            event_id=event_id,
            processed_at=datetime.utcnow(),
            payload_hash=self.compute_payload_hash(payload)
        )

        # Add to cache
        self._cache[event_id] = event

        # Trim cache if needed
        if len(self._cache) > self.cache_size:
            oldest_keys = sorted(
                self._cache.keys(),
                key=lambda k: self._cache[k].processed_at
            )[:len(self._cache) - self.cache_size]
            for k in oldest_keys:
                del self._cache[k]

        # Persist to DB
        if self.persist_to_db and self._db_conn:
            try:
                cur = self._db_conn.cursor()
                cur.execute(
                    """
                    INSERT INTO event_processed_log (event_id, event_type, payload_hash)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (event_id) DO NOTHING
                    """,
                    (event_id, payload.get("event_type", "unknown"), event.payload_hash)
                )
                self._db_conn.commit()
                cur.close()
            except Exception as e:
                logger.warning(f"DB persist failed: {e}")

    def set_connection(self, conn) -> None:
        """Set database connection for persistence."""
        self._db_conn = conn


# =============================================================================
# DEAD LETTER QUEUE
# =============================================================================

class DeadLetterQueue:
    """
    Stores failed events for retry.

    Events that fail processing are stored in DLQ with exponential backoff.
    After max retries, events are marked as "dead" for manual review.
    """

    def __init__(self, conn, max_retries: int = 5):
        self.conn = conn
        self.max_retries = max_retries

    def enqueue(
        self,
        event_id: str,
        event_type: str,
        channel: str,
        payload: Dict[str, Any],
        error: str
    ) -> Optional[int]:
        """Add failed event to DLQ."""
        if not self.conn:
            logger.warning("DLQ: No database connection")
            return None

        try:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT dlq_enqueue(%s, %s, %s, %s, %s)",
                (event_id, event_type, channel, json.dumps(payload), error)
            )
            result = cur.fetchone()
            self.conn.commit()
            cur.close()

            dlq_id = result[0] if result else None
            logger.warning(f"DLQ: Enqueued event {event_id} (id={dlq_id})")
            return dlq_id

        except Exception as e:
            logger.error(f"DLQ enqueue failed: {e}")
            return None

    def get_pending_for_retry(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get events ready for retry."""
        if not self.conn:
            return []

        try:
            cur = self.conn.cursor()
            cur.execute(
                """
                SELECT id, event_id, event_type, channel, payload, error_count
                FROM event_dead_letter_queue
                WHERE status = 'pending' AND retry_after <= NOW()
                ORDER BY error_count ASC, first_failed_at ASC
                LIMIT %s
                """,
                (limit,)
            )
            rows = cur.fetchall()
            cur.close()

            return [
                {
                    "id": row[0],
                    "event_id": row[1],
                    "event_type": row[2],
                    "channel": row[3],
                    "payload": row[4],
                    "error_count": row[5]
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"DLQ fetch failed: {e}")
            return []


# =============================================================================
# POSTGRESQL NOTIFY SENSOR BASE
# =============================================================================

class PostgresNotifySensorBase(BaseSensorOperator):
    """
    Base sensor for PostgreSQL NOTIFY with Circuit Breaker.

    Listens to a PostgreSQL channel for events. If NOTIFY fails repeatedly,
    falls back to polling mode via Circuit Breaker pattern.
    """

    template_fields = ('channel', 'fallback_table')

    @apply_defaults
    def __init__(
        self,
        channel: str,
        fallback_table: str,
        fallback_date_column: str = 'time',
        max_staleness_minutes: int = 10,
        poke_interval: int = 30,
        timeout: int = 300,
        mode: str = 'poke',
        # Circuit breaker config
        max_failures: int = 3,
        circuit_reset_seconds: int = 300,
        # Connection
        postgres_conn_id: str = 'timescale_conn',
        **kwargs
    ):
        super().__init__(
            poke_interval=poke_interval,
            timeout=timeout,
            mode=mode,
            **kwargs
        )
        self.channel = channel
        self.fallback_table = fallback_table
        self.fallback_date_column = fallback_date_column
        self.max_staleness_minutes = max_staleness_minutes
        self.postgres_conn_id = postgres_conn_id

        # Initialize circuit breaker
        self._circuit = CircuitBreaker(
            max_failures=max_failures,
            reset_timeout_seconds=circuit_reset_seconds
        )

        # Initialize idempotent processor
        self._processor = IdempotentProcessor()

        # Connection state
        self._listen_conn: Optional[extensions.connection] = None
        self._last_seen_time: Optional[datetime] = None

    def _get_connection(self):
        """Get database connection from Airflow."""
        from airflow.hooks.postgres_hook import PostgresHook
        hook = PostgresHook(postgres_conn_id=self.postgres_conn_id)
        return hook.get_conn()

    def _get_listen_connection(self) -> Optional[extensions.connection]:
        """Get dedicated connection for LISTEN."""
        if self._listen_conn is None or self._listen_conn.closed:
            try:
                self._listen_conn = self._get_connection()
                self._listen_conn.set_isolation_level(
                    extensions.ISOLATION_LEVEL_AUTOCOMMIT
                )
                cur = self._listen_conn.cursor()
                cur.execute(f"LISTEN {self.channel};")
                cur.close()
                logger.info(f"Listening on channel: {self.channel}")
            except Exception as e:
                logger.error(f"Failed to establish LISTEN connection: {e}")
                self._listen_conn = None
        return self._listen_conn

    def _check_notify(self, timeout: float = 1.0) -> List[Dict[str, Any]]:
        """Check for NOTIFY events."""
        conn = self._get_listen_connection()
        if not conn:
            raise RuntimeError("No LISTEN connection available")

        events = []
        try:
            if select.select([conn], [], [], timeout) == ([], [], []):
                return []

            conn.poll()
            while conn.notifies:
                notify = conn.notifies.pop(0)
                try:
                    payload = json.loads(notify.payload)
                    events.append(payload)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in NOTIFY payload: {notify.payload}")

        except Exception as e:
            logger.error(f"Error checking NOTIFY: {e}")
            self._circuit.record_failure()
            raise

        return events

    def _check_polling(self) -> Tuple[bool, Optional[datetime]]:
        """Fallback polling check."""
        conn = None
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute(f"""
                SELECT MAX({self.fallback_date_column}) as latest_time
                FROM {self.fallback_table}
            """)

            result = cur.fetchone()
            latest_time = result[0] if result else None

            cur.close()
            conn.close()

            if latest_time is None:
                return False, None

            # Check staleness
            now = datetime.utcnow()
            if hasattr(latest_time, 'replace'):
                staleness = (now - latest_time.replace(tzinfo=None)).total_seconds() / 60
            else:
                staleness = 0

            if staleness > self.max_staleness_minutes:
                return False, latest_time

            # Check if new data
            if self._last_seen_time and latest_time <= self._last_seen_time:
                return False, latest_time

            return True, latest_time

        except Exception as e:
            logger.error(f"Polling check failed: {e}")
            if conn:
                conn.close()
            return False, None

    def poke(self, context) -> bool:
        """
        Main sensor logic with Circuit Breaker.

        Uses NOTIFY when circuit is closed, falls back to polling when open.
        """
        ti = context.get('ti')

        # Check circuit breaker state
        if self._circuit.should_allow_request():
            try:
                # Try NOTIFY
                events = self._check_notify(timeout=self.poke_interval * 0.8)

                if events:
                    # Process events with idempotency
                    for event in events:
                        event_id = self._processor.compute_event_id(event)

                        if self._processor.is_processed(event_id):
                            logger.debug(f"Skipping duplicate event: {event_id}")
                            continue

                        # Mark as processed
                        self._processor.mark_processed(event_id, event)

                        # Store latest time
                        if 'time' in event:
                            self._last_seen_time = datetime.fromisoformat(
                                event['time'].replace('Z', '+00:00')
                            )

                        # Push to XCom
                        if ti:
                            ti.xcom_push(key='notify_event', value=event)
                            ti.xcom_push(key='detected_time', value=event.get('time'))

                        logger.info(f"NOTIFY event received: {event.get('event_type')}")
                        self._circuit.record_success()
                        return True

                # No events, but no failure either
                return False

            except Exception as e:
                logger.warning(f"NOTIFY failed: {e}, circuit may open")
                # Circuit breaker handles failure counting

        # Circuit is open - use polling fallback
        if self._circuit.is_using_fallback:
            logger.debug("Using polling fallback (circuit open)")
            has_new_data, latest_time = self._check_polling()

            if has_new_data:
                self._last_seen_time = latest_time

                if ti:
                    ti.xcom_push(key='detected_time', value=str(latest_time))
                    ti.xcom_push(key='source', value='polling_fallback')

                logger.info(f"Polling detected new data at {latest_time}")
                return True

        return False

    def cleanup(self) -> None:
        """Clean up connections."""
        if self._listen_conn:
            try:
                cur = self._listen_conn.cursor()
                cur.execute(f"UNLISTEN {self.channel};")
                cur.close()
                self._listen_conn.close()
            except:
                pass
            self._listen_conn = None


# =============================================================================
# OHLCV BAR SENSOR
# =============================================================================

class OHLCVBarSensor(PostgresNotifySensorBase):
    """
    Sensor that waits for new OHLCV bar via PostgreSQL NOTIFY.

    Listens to 'ohlcv_updates' channel for new bar events.
    Falls back to polling if NOTIFY fails (circuit breaker).

    Example:
        sensor = OHLCVBarSensor(
            task_id='wait_for_ohlcv',
            symbol='USD/COP',
            timeout=300,
        )
        sensor >> task_calculate_features
    """

    @apply_defaults
    def __init__(
        self,
        symbol: str = 'USD/COP',
        channel: str = 'ohlcv_updates',
        fallback_table: str = 'usdcop_m5_ohlcv',
        **kwargs
    ):
        super().__init__(
            channel=channel,
            fallback_table=fallback_table,
            fallback_date_column='time',
            **kwargs
        )
        self.symbol = symbol

    def poke(self, context) -> bool:
        """Check for new OHLCV bar."""
        result = super().poke(context)

        if result:
            ti = context.get('ti')
            if ti:
                ti.xcom_push(key='symbol', value=self.symbol)

        return result


# =============================================================================
# FEATURE READY SENSOR
# =============================================================================

class FeatureReadySensor(PostgresNotifySensorBase):
    """
    Sensor that waits for features to be ready via PostgreSQL NOTIFY.

    v6.0.0: Default channel changed to 'features_ready' and fallback table to
    'inference_ready_nrt' (pre-normalized by L1 DAG).

    Listens to NOTIFY channel for feature completion events.
    Falls back to polling on fallback_table if NOTIFY fails.

    Example:
        sensor = FeatureReadySensor(
            task_id='wait_for_features',
            timeout=300,
        )
        sensor >> task_run_inference
    """

    # Critical features that must be present (used for inference_features_5m fallback)
    CRITICAL_FEATURES = [
        'log_ret_5m', 'log_ret_1h', 'rsi_9',
        'dxy_z', 'vix_z', 'rate_spread',
    ]

    @apply_defaults
    def __init__(
        self,
        require_complete: bool = True,
        channel: str = 'features_ready',
        fallback_table: str = 'inference_ready_nrt',
        fallback_date_column: str = 'timestamp',
        **kwargs
    ):
        super().__init__(
            channel=channel,
            fallback_table=fallback_table,
            fallback_date_column=fallback_date_column,
            **kwargs
        )
        self.require_complete = require_complete

    def _check_polling(self) -> Tuple[bool, Optional[datetime]]:
        """Override to check feature freshness.

        For inference_ready_nrt: checks that FLOAT[] features are present and fresh.
        For inference_features_5m (legacy): checks individual columns.
        """
        conn = None
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            date_col = self.fallback_date_column

            if self.fallback_table == 'inference_ready_nrt':
                # For inference_ready_nrt: check features FLOAT[] is not null
                where_clause = "WHERE features IS NOT NULL" if self.require_complete else ""
            elif self.require_complete:
                # Legacy: check individual feature columns
                checks = " AND ".join(
                    f"{feat} IS NOT NULL" for feat in self.CRITICAL_FEATURES
                )
                where_clause = f"WHERE {checks}"
            else:
                where_clause = ""

            cur.execute(f"""
                SELECT MAX({date_col}) as latest_time
                FROM {self.fallback_table}
                {where_clause}
            """)

            result = cur.fetchone()
            latest_time = result[0] if result else None

            cur.close()
            conn.close()

            if latest_time is None:
                return False, None

            # Check staleness
            now = datetime.utcnow()
            if hasattr(latest_time, 'replace'):
                staleness = (now - latest_time.replace(tzinfo=None)).total_seconds() / 60
            else:
                staleness = 0

            if staleness > self.max_staleness_minutes:
                return False, latest_time

            # Check if new
            if self._last_seen_time and latest_time <= self._last_seen_time:
                return False, latest_time

            return True, latest_time

        except Exception as e:
            logger.error(f"Feature polling failed: {e}")
            if conn:
                conn.close()
            return False, None


# =============================================================================
# HEARTBEAT MONITOR
# =============================================================================

class HeartbeatMonitor:
    """
    Monitors NOTIFY system health via heartbeat channel.

    Sends periodic heartbeats and verifies receipt to detect NOTIFY failures
    before they impact production sensors.
    """

    HEARTBEAT_CHANNEL = "heartbeat"
    HEARTBEAT_INTERVAL = 60  # seconds
    MAX_MISSED_HEARTBEATS = 3

    def __init__(self, conn):
        self.conn = conn
        self.missed_heartbeats = 0
        self.last_heartbeat_received: Optional[datetime] = None
        self.is_healthy = True

    def send_heartbeat(self) -> bool:
        """Send a heartbeat via pg_notify."""
        try:
            cur = self.conn.cursor()
            cur.execute(f"SELECT emit_heartbeat('{self.HEARTBEAT_CHANNEL}')")
            self.conn.commit()
            cur.close()
            return True
        except Exception as e:
            logger.error(f"Failed to send heartbeat: {e}")
            return False

    def check_heartbeat(self, timeout: float = 5.0) -> bool:
        """Check if heartbeat was received."""
        try:
            self.conn.set_isolation_level(extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            cur = self.conn.cursor()
            cur.execute(f"LISTEN {self.HEARTBEAT_CHANNEL};")
            cur.close()

            # Send heartbeat
            if not self.send_heartbeat():
                return False

            # Wait for receipt
            if select.select([self.conn], [], [], timeout) == ([], [], []):
                self.missed_heartbeats += 1
                logger.warning(f"Heartbeat not received (missed: {self.missed_heartbeats})")
                self.is_healthy = self.missed_heartbeats < self.MAX_MISSED_HEARTBEATS
                return False

            self.conn.poll()
            if self.conn.notifies:
                self.conn.notifies.pop(0)
                self.missed_heartbeats = 0
                self.last_heartbeat_received = datetime.utcnow()
                self.is_healthy = True
                return True

            return False

        except Exception as e:
            logger.error(f"Heartbeat check failed: {e}")
            self.is_healthy = False
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return {
            "is_healthy": self.is_healthy,
            "missed_heartbeats": self.missed_heartbeats,
            "last_heartbeat_received": str(self.last_heartbeat_received) if self.last_heartbeat_received else None,
            "max_allowed_missed": self.MAX_MISSED_HEARTBEATS,
        }


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# Aliases for backward compatibility with existing DAGs
NewOHLCVBarSensor = OHLCVBarSensor
NewFeatureBarSensor = FeatureReadySensor
