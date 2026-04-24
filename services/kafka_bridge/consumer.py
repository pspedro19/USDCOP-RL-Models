"""
Kafka consumer for USDCOP H5 trading signals.

Subscribes to topic `signals.h5` with consumer group `signalbridge-consumer`,
prints each message with a timestamp, and appends to /data/consumed.log.

Supports `--count N` to consume up to N messages and exit (for demo / tests),
plus `--timeout SECONDS` to bound waiting in demo mode.

Contract:
  - Broker: env KAFKA_BROKER (default redpanda:9092)
  - Topic: signals.h5
  - Group: signalbridge-consumer
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kafka import KafkaConsumer
from kafka.errors import KafkaError, NoBrokersAvailable

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "redpanda:9092")
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "signals.h5")
CONSUMER_GROUP = os.environ.get("CONSUMER_GROUP", "signalbridge-consumer")
LOG_DIR = Path(os.environ.get("STATE_DIR", "/data"))
LOG_FILE = LOG_DIR / "consumed.log"

MAX_CONNECT_ATTEMPTS = int(os.environ.get("MAX_CONNECT_ATTEMPTS", "30"))
CONNECT_RETRY_SEC = int(os.environ.get("CONNECT_RETRY_SEC", "10"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [consumer] %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("kafka_bridge.consumer")

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_shutdown = False


def _handle_sigterm(signum, frame):  # pragma: no cover - signal path
    global _shutdown
    log.info("received signal %s, shutting down gracefully", signum)
    _shutdown = True


signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)


# ---------------------------------------------------------------------------
# Log file helper
# ---------------------------------------------------------------------------


def _ensure_log_dir() -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        log.warning("could not create log dir %s: %s", LOG_DIR, exc)


def append_consumed(line: str) -> None:
    _ensure_log_dir()
    try:
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(line.rstrip("\n") + "\n")
    except Exception as exc:
        log.warning("could not append to %s: %s", LOG_FILE, exc)


# ---------------------------------------------------------------------------
# Kafka consumer connection with bounded retries
# ---------------------------------------------------------------------------


def build_consumer(consumer_timeout_ms: int | None = None) -> KafkaConsumer:
    attempt = 0
    last_exc: Exception | None = None
    while attempt < MAX_CONNECT_ATTEMPTS and not _shutdown:
        attempt += 1
        try:
            kwargs: dict[str, Any] = dict(
                bootstrap_servers=KAFKA_BROKER,
                group_id=CONSUMER_GROUP,
                auto_offset_reset=os.environ.get("AUTO_OFFSET_RESET", "earliest"),
                enable_auto_commit=True,
                value_deserializer=_safe_deserialize,
                client_id="usdcop-kafka-bridge-consumer",
            )
            if consumer_timeout_ms is not None:
                kwargs["consumer_timeout_ms"] = consumer_timeout_ms
            consumer = KafkaConsumer(KAFKA_TOPIC, **kwargs)
            log.info(
                "connected to Kafka broker=%s topic=%s group=%s (attempt %d)",
                KAFKA_BROKER,
                KAFKA_TOPIC,
                CONSUMER_GROUP,
                attempt,
            )
            return consumer
        except (NoBrokersAvailable, KafkaError, OSError) as exc:
            last_exc = exc
            log.warning(
                "Kafka unreachable (attempt %d/%d): %s — retrying in %ds",
                attempt,
                MAX_CONNECT_ATTEMPTS,
                exc,
                CONNECT_RETRY_SEC,
            )
            _sleep_interruptible(CONNECT_RETRY_SEC)

    msg = f"could not connect to Kafka {KAFKA_BROKER} after {MAX_CONNECT_ATTEMPTS} attempts"
    if last_exc is not None:
        msg += f": {last_exc}"
    raise RuntimeError(msg)


def _safe_deserialize(raw: bytes) -> Any:
    if raw is None:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        try:
            return {"_raw": raw.decode("utf-8", errors="replace")}
        except Exception:
            return {"_raw": repr(raw)}


def _sleep_interruptible(seconds: float) -> None:
    end = time.time() + seconds
    while not _shutdown and time.time() < end:
        time.sleep(min(0.5, max(0.0, end - time.time())))


# ---------------------------------------------------------------------------
# Message handler
# ---------------------------------------------------------------------------


def format_record(msg: Any) -> str:
    now = datetime.now(tz=timezone.utc).isoformat()
    try:
        payload = msg.value if hasattr(msg, "value") else msg
        return (
            f"{now} partition={getattr(msg, 'partition', '-')} "
            f"offset={getattr(msg, 'offset', '-')} "
            f"value={json.dumps(payload, default=str)}"
        )
    except Exception as exc:
        return f"{now} format_error={exc} raw={msg!r}"


# ---------------------------------------------------------------------------
# Main consume loop
# ---------------------------------------------------------------------------


def consume(count: int | None, timeout_sec: int | None) -> int:
    consumer_timeout_ms: int | None = None
    # Bounded mode: break out of poll() when idle for timeout_sec
    if count is not None and timeout_sec is not None:
        consumer_timeout_ms = max(1, timeout_sec) * 1000

    try:
        consumer = build_consumer(consumer_timeout_ms=consumer_timeout_ms)
    except RuntimeError as exc:
        log.error("%s", exc)
        return 2

    deadline = time.time() + timeout_sec if timeout_sec is not None else None
    received = 0

    try:
        for msg in consumer:
            if _shutdown:
                break
            try:
                line = format_record(msg)
                log.info("received %s", line)
                append_consumed(line)
                received += 1
            except Exception as exc:
                log.warning("error handling message: %s", exc)

            if count is not None and received >= count:
                log.info("count target reached (%d)", received)
                break
            if deadline is not None and time.time() >= deadline:
                log.info("timeout reached after %d message(s)", received)
                break
    except StopIteration:
        # Raised by kafka-python when consumer_timeout_ms elapses
        log.info("consumer idle timeout reached after %d message(s)", received)
    except Exception as exc:
        log.warning("consumer loop error: %s", exc)
    finally:
        try:
            consumer.close(autocommit=True)
        except Exception:
            pass

    if count is not None:
        log.info("bounded consume finished: %d/%s", received, count)
        # Exit 0 even if fewer messages arrived; the test harness checks logs.
        return 0

    log.info("consumer exiting cleanly (received %d)", received)
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="USDCOP H5 Kafka consumer")
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="consume up to N messages and exit",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="max seconds to wait (used with --count)",
    )
    args = parser.parse_args()

    log.info(
        "kafka_bridge.consumer starting: broker=%s topic=%s group=%s count=%s timeout=%s",
        KAFKA_BROKER,
        KAFKA_TOPIC,
        CONSUMER_GROUP,
        args.count,
        args.timeout,
    )

    return consume(count=args.count, timeout_sec=args.timeout)


if __name__ == "__main__":
    sys.exit(main())
