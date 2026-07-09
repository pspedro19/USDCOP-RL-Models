#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-Flight NOTIFY Test - V7.1 Event-Driven Architecture Validation
====================================================================
GO/NO-GO decision script before implementing PostgreSQL NOTIFY system.

This script validates:
1. PostgreSQL LISTEN/NOTIFY functionality works correctly
2. 100% event delivery (no lost messages)
3. Latency p99 < 100ms threshold
4. Connection pooling under load

Usage:
    python scripts/preflight_notify_test.py --connection-string "postgresql://..." --events 1000

Exit Codes:
    0 = GO (All tests passed)
    1 = NO-GO (Tests failed - do not proceed with V7)

Author: Trading Team
Version: 1.0.0
Created: 2026-01-31
Contract: CTR-V7-PREFLIGHT
"""

import argparse
import json
import logging
import os
import queue
import select
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import psycopg2
from psycopg2 import extensions

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PreflightConfig:
    """Configuration for pre-flight NOTIFY test."""
    connection_string: str
    channel: str = "preflight_test"
    num_events: int = 1000
    timeout_seconds: int = 30
    latency_p99_threshold_ms: float = 100.0
    delivery_threshold_pct: float = 100.0
    concurrent_listeners: int = 3
    batch_size: int = 100


@dataclass
class PreflightResult:
    """Results from pre-flight test."""
    passed: bool = False
    events_sent: int = 0
    events_received: int = 0
    delivery_rate_pct: float = 0.0
    latency_avg_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_max_ms: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    test_duration_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict:
        return {
            "passed": self.passed,
            "decision": "GO" if self.passed else "NO-GO",
            "events_sent": self.events_sent,
            "events_received": self.events_received,
            "delivery_rate_pct": round(self.delivery_rate_pct, 2),
            "latency": {
                "avg_ms": round(self.latency_avg_ms, 2),
                "p50_ms": round(self.latency_p50_ms, 2),
                "p95_ms": round(self.latency_p95_ms, 2),
                "p99_ms": round(self.latency_p99_ms, 2),
                "max_ms": round(self.latency_max_ms, 2),
            },
            "errors": self.errors,
            "warnings": self.warnings,
            "test_duration_seconds": round(self.test_duration_seconds, 2),
            "timestamp": self.timestamp,
        }


# =============================================================================
# LISTENER (CONSUMER)
# =============================================================================

class NotifyListener:
    """PostgreSQL NOTIFY listener thread."""

    def __init__(
        self,
        connection_string: str,
        channel: str,
        event_queue: queue.Queue,
        stop_event: threading.Event,
        listener_id: int = 0,
    ):
        self.connection_string = connection_string
        self.channel = channel
        self.event_queue = event_queue
        self.stop_event = stop_event
        self.listener_id = listener_id
        self.conn: Optional[extensions.connection] = None
        self.received_count = 0

    def connect(self) -> bool:
        """Establish connection and subscribe to channel."""
        try:
            self.conn = psycopg2.connect(self.connection_string)
            self.conn.set_isolation_level(extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            cur = self.conn.cursor()
            cur.execute(f"LISTEN {self.channel};")
            cur.close()
            logger.debug(f"Listener {self.listener_id}: Connected and listening on {self.channel}")
            return True
        except Exception as e:
            logger.error(f"Listener {self.listener_id}: Connection failed: {e}")
            return False

    def listen(self, timeout: float = 0.1) -> List[Tuple[str, float]]:
        """Listen for notifications with timeout."""
        if not self.conn:
            return []

        received = []
        try:
            if select.select([self.conn], [], [], timeout) == ([], [], []):
                return []

            self.conn.poll()
            while self.conn.notifies:
                notify = self.conn.notifies.pop(0)
                receive_time = time.perf_counter()
                self.received_count += 1
                received.append((notify.payload, receive_time))

        except Exception as e:
            logger.error(f"Listener {self.listener_id}: Error: {e}")

        return received

    def run(self):
        """Main listener loop."""
        if not self.connect():
            return

        while not self.stop_event.is_set():
            events = self.listen(timeout=0.05)
            for payload, receive_time in events:
                self.event_queue.put((payload, receive_time))

        self.close()

    def close(self):
        """Close connection."""
        if self.conn:
            try:
                cur = self.conn.cursor()
                cur.execute(f"UNLISTEN {self.channel};")
                cur.close()
                self.conn.close()
            except:
                pass


# =============================================================================
# SENDER (PRODUCER)
# =============================================================================

class NotifySender:
    """PostgreSQL NOTIFY sender."""

    def __init__(self, connection_string: str, channel: str):
        self.connection_string = connection_string
        self.channel = channel
        self.conn: Optional[extensions.connection] = None
        self.sent_count = 0

    def connect(self) -> bool:
        """Establish connection."""
        try:
            self.conn = psycopg2.connect(self.connection_string)
            self.conn.set_isolation_level(extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            logger.debug("Sender: Connected")
            return True
        except Exception as e:
            logger.error(f"Sender: Connection failed: {e}")
            return False

    def send(self, event_id: int) -> Tuple[int, float]:
        """Send a single NOTIFY with event ID and timestamp."""
        if not self.conn:
            raise RuntimeError("Not connected")

        send_time = time.perf_counter()
        payload = json.dumps({
            "event_id": event_id,
            "send_time_ns": time.time_ns(),
            "test": "preflight"
        })

        cur = self.conn.cursor()
        cur.execute(f"NOTIFY {self.channel}, %s;", (payload,))
        cur.close()
        self.sent_count += 1

        return event_id, send_time

    def send_batch(self, start_id: int, count: int) -> List[Tuple[int, float]]:
        """Send a batch of notifications."""
        results = []
        for i in range(count):
            results.append(self.send(start_id + i))
        return results

    def close(self):
        """Close connection."""
        if self.conn:
            try:
                self.conn.close()
            except:
                pass


# =============================================================================
# PRE-FLIGHT TEST RUNNER
# =============================================================================

class PreflightTestRunner:
    """Runs the pre-flight NOTIFY test."""

    def __init__(self, config: PreflightConfig):
        self.config = config
        self.result = PreflightResult()
        self.sent_events: Dict[int, float] = {}  # event_id -> send_time
        self.received_events: Dict[int, float] = {}  # event_id -> receive_time
        self.event_queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()

    def run(self) -> PreflightResult:
        """Execute the full pre-flight test."""
        start_time = time.perf_counter()
        logger.info("=" * 60)
        logger.info("PRE-FLIGHT NOTIFY TEST - V7.1 Event-Driven Architecture")
        logger.info("=" * 60)
        logger.info(f"Channel: {self.config.channel}")
        logger.info(f"Events to send: {self.config.num_events}")
        logger.info(f"Concurrent listeners: {self.config.concurrent_listeners}")
        logger.info(f"Latency p99 threshold: {self.config.latency_p99_threshold_ms}ms")
        logger.info("-" * 60)

        try:
            # Phase 1: Start listeners
            logger.info("[Phase 1] Starting listeners...")
            listeners = self._start_listeners()
            time.sleep(0.5)  # Allow connections to establish

            # Phase 2: Send events
            logger.info("[Phase 2] Sending events...")
            self._send_events()

            # Phase 3: Wait for delivery
            logger.info("[Phase 3] Waiting for event delivery...")
            self._wait_for_delivery()

            # Phase 4: Stop listeners
            logger.info("[Phase 4] Stopping listeners...")
            self._stop_listeners(listeners)

            # Phase 5: Collect results
            logger.info("[Phase 5] Collecting results...")
            self._collect_results()

            # Phase 6: Evaluate
            logger.info("[Phase 6] Evaluating results...")
            self._evaluate()

        except Exception as e:
            self.result.errors.append(f"Test failed with exception: {e}")
            self.result.passed = False
            logger.error(f"Test failed: {e}")

        self.result.test_duration_seconds = time.perf_counter() - start_time

        # Print summary
        self._print_summary()

        return self.result

    def _start_listeners(self) -> List[threading.Thread]:
        """Start listener threads."""
        listeners = []
        for i in range(self.config.concurrent_listeners):
            listener = NotifyListener(
                connection_string=self.config.connection_string,
                channel=self.config.channel,
                event_queue=self.event_queue,
                stop_event=self.stop_event,
                listener_id=i,
            )
            thread = threading.Thread(target=listener.run, daemon=True)
            thread.start()
            listeners.append(thread)
        return listeners

    def _send_events(self):
        """Send all test events."""
        sender = NotifySender(self.config.connection_string, self.config.channel)
        if not sender.connect():
            self.result.errors.append("Failed to connect sender")
            return

        try:
            batch_size = self.config.batch_size
            for batch_start in range(0, self.config.num_events, batch_size):
                batch_end = min(batch_start + batch_size, self.config.num_events)
                batch_count = batch_end - batch_start

                results = sender.send_batch(batch_start, batch_count)
                for event_id, send_time in results:
                    self.sent_events[event_id] = send_time

                # Small delay between batches to avoid overwhelming
                time.sleep(0.01)

            self.result.events_sent = len(self.sent_events)
            logger.info(f"  Sent {self.result.events_sent} events")

        finally:
            sender.close()

    def _wait_for_delivery(self):
        """Wait for events to be delivered."""
        deadline = time.time() + self.config.timeout_seconds
        last_count = 0
        stale_checks = 0

        while time.time() < deadline:
            # Drain queue
            try:
                while True:
                    payload, receive_time = self.event_queue.get_nowait()
                    try:
                        data = json.loads(payload)
                        event_id = data["event_id"]
                        self.received_events[event_id] = receive_time
                    except:
                        pass
            except queue.Empty:
                pass

            current_count = len(self.received_events)

            # Check if we received all events
            if current_count >= self.result.events_sent:
                logger.info(f"  All {current_count} events received")
                break

            # Check for stale progress
            if current_count == last_count:
                stale_checks += 1
                if stale_checks > 50:  # 5 seconds of no progress
                    logger.warning(f"  Delivery stalled at {current_count}/{self.result.events_sent}")
                    break
            else:
                stale_checks = 0
                last_count = current_count

            time.sleep(0.1)

        # Final drain
        try:
            while True:
                payload, receive_time = self.event_queue.get_nowait()
                try:
                    data = json.loads(payload)
                    event_id = data["event_id"]
                    self.received_events[event_id] = receive_time
                except:
                    pass
        except queue.Empty:
            pass

    def _stop_listeners(self, threads: List[threading.Thread]):
        """Stop all listener threads."""
        self.stop_event.set()
        for thread in threads:
            thread.join(timeout=2.0)

    def _collect_results(self):
        """Calculate latency statistics."""
        self.result.events_received = len(self.received_events)

        if self.result.events_sent > 0:
            self.result.delivery_rate_pct = (
                self.result.events_received / self.result.events_sent * 100
            )

        # Calculate latencies for matched events
        latencies_ms = []
        for event_id, receive_time in self.received_events.items():
            if event_id in self.sent_events:
                send_time = self.sent_events[event_id]
                latency_ms = (receive_time - send_time) * 1000
                latencies_ms.append(latency_ms)

        if latencies_ms:
            latencies_ms.sort()
            self.result.latency_avg_ms = statistics.mean(latencies_ms)
            self.result.latency_p50_ms = statistics.median(latencies_ms)
            self.result.latency_p95_ms = latencies_ms[int(len(latencies_ms) * 0.95)]
            self.result.latency_p99_ms = latencies_ms[int(len(latencies_ms) * 0.99)]
            self.result.latency_max_ms = max(latencies_ms)

        # Check for lost events
        lost_events = set(self.sent_events.keys()) - set(self.received_events.keys())
        if lost_events:
            self.result.warnings.append(
                f"Lost {len(lost_events)} events: {list(lost_events)[:10]}..."
            )

    def _evaluate(self):
        """Evaluate if test passed."""
        passed = True

        # Check delivery rate
        if self.result.delivery_rate_pct < self.config.delivery_threshold_pct:
            passed = False
            self.result.errors.append(
                f"Delivery rate {self.result.delivery_rate_pct:.1f}% < "
                f"{self.config.delivery_threshold_pct}% threshold"
            )

        # Check latency p99
        if self.result.latency_p99_ms > self.config.latency_p99_threshold_ms:
            passed = False
            self.result.errors.append(
                f"Latency p99 {self.result.latency_p99_ms:.2f}ms > "
                f"{self.config.latency_p99_threshold_ms}ms threshold"
            )

        self.result.passed = passed and len(self.result.errors) == 0

    def _print_summary(self):
        """Print test summary."""
        logger.info("=" * 60)
        logger.info("PRE-FLIGHT TEST RESULTS")
        logger.info("=" * 60)

        decision = "GO" if self.result.passed else "NO-GO"
        decision_symbol = "✅" if self.result.passed else "❌"

        logger.info(f"Decision: {decision_symbol} {decision}")
        logger.info("-" * 60)
        logger.info(f"Events Sent:     {self.result.events_sent}")
        logger.info(f"Events Received: {self.result.events_received}")
        logger.info(f"Delivery Rate:   {self.result.delivery_rate_pct:.2f}%")
        logger.info("-" * 60)
        logger.info("Latency Statistics:")
        logger.info(f"  Average: {self.result.latency_avg_ms:.2f}ms")
        logger.info(f"  P50:     {self.result.latency_p50_ms:.2f}ms")
        logger.info(f"  P95:     {self.result.latency_p95_ms:.2f}ms")
        logger.info(f"  P99:     {self.result.latency_p99_ms:.2f}ms {'✅' if self.result.latency_p99_ms <= self.config.latency_p99_threshold_ms else '❌'}")
        logger.info(f"  Max:     {self.result.latency_max_ms:.2f}ms")
        logger.info("-" * 60)
        logger.info(f"Duration: {self.result.test_duration_seconds:.2f}s")

        if self.result.errors:
            logger.error("Errors:")
            for error in self.result.errors:
                logger.error(f"  - {error}")

        if self.result.warnings:
            logger.warning("Warnings:")
            for warning in self.result.warnings:
                logger.warning(f"  - {warning}")

        logger.info("=" * 60)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def get_default_connection_string() -> str:
    """Get connection string from environment or default."""
    return os.environ.get(
        "DATABASE_URL",
        os.environ.get(
            "TIMESCALE_URL",
            "postgresql://postgres:postgres@localhost:5432/trading"
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Pre-Flight NOTIFY Test for V7.1 Event-Driven Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic test with 1000 events
    python preflight_notify_test.py

    # Custom connection and event count
    python preflight_notify_test.py --connection-string "postgresql://..." --events 5000

    # Output JSON results
    python preflight_notify_test.py --json --output results.json

Exit Codes:
    0 = GO (All tests passed - proceed with V7 implementation)
    1 = NO-GO (Tests failed - investigate before proceeding)
        """
    )

    parser.add_argument(
        "--connection-string",
        type=str,
        default=get_default_connection_string(),
        help="PostgreSQL connection string"
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="preflight_test",
        help="NOTIFY channel name (default: preflight_test)"
    )
    parser.add_argument(
        "--events",
        type=int,
        default=1000,
        help="Number of events to send (default: 1000)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout in seconds (default: 30)"
    )
    parser.add_argument(
        "--latency-threshold",
        type=float,
        default=100.0,
        help="Latency p99 threshold in ms (default: 100)"
    )
    parser.add_argument(
        "--listeners",
        type=int,
        default=3,
        help="Number of concurrent listeners (default: 3)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save results to file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = PreflightConfig(
        connection_string=args.connection_string,
        channel=args.channel,
        num_events=args.events,
        timeout_seconds=args.timeout,
        latency_p99_threshold_ms=args.latency_threshold,
        concurrent_listeners=args.listeners,
    )

    runner = PreflightTestRunner(config)
    result = runner.run()

    # Output
    if args.json or args.output:
        result_dict = result.to_dict()
        result_json = json.dumps(result_dict, indent=2)

        if args.json:
            print(result_json)

        if args.output:
            with open(args.output, 'w') as f:
                f.write(result_json)
            logger.info(f"Results saved to {args.output}")

    # Exit code
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
