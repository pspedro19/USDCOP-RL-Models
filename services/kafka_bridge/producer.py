"""
Kafka producer for USDCOP H5 trading signals.

Polls the `forecast_h5_signals` PostgreSQL table every 60 seconds for new signals
(tracked via /data/published.txt, storing the last published signal_id) and
publishes them to the Kafka topic `signals.h5`.

Demo mode (--demo) publishes 3 synthetic signals and exits.

Contract:
  - Broker: env KAFKA_BROKER (default redpanda:9092)
  - Topic: signals.h5
  - DB: env DATABASE_URL (fallback postgresql://admin:admin123@postgres:5432/usdcop_trading)
  - Message: JSON { week, direction, confidence, ensemble_return, skip_trade,
                    hard_stop_pct, take_profit_pct, adjusted_leverage, timestamp }
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

try:
    from dateutil import tz as _tz
    _COT = _tz.gettz("America/Bogota")
except Exception:  # pragma: no cover
    _COT = timezone.utc

try:
    import psycopg2
    import psycopg2.extras
except Exception:  # pragma: no cover - module still imports in demo-only env
    psycopg2 = None  # type: ignore[assignment]

from kafka import KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "redpanda:9092")
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "signals.h5")
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://admin:admin123@postgres:5432/usdcop_trading",
)
POLL_INTERVAL_SEC = int(os.environ.get("POLL_INTERVAL_SEC", "60"))
STATE_DIR = Path(os.environ.get("STATE_DIR", "/data"))
STATE_FILE = STATE_DIR / "published.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [producer] %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("kafka_bridge.producer")

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
# State persistence
# ---------------------------------------------------------------------------


def _ensure_state_dir() -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        log.warning("could not create state dir %s: %s", STATE_DIR, exc)


def load_last_published_id() -> int:
    _ensure_state_dir()
    if not STATE_FILE.exists():
        return 0
    try:
        text = STATE_FILE.read_text().strip()
        return int(text) if text else 0
    except Exception as exc:
        log.warning("could not parse state file %s: %s — treating as 0", STATE_FILE, exc)
        return 0


def save_last_published_id(signal_id: int) -> None:
    _ensure_state_dir()
    try:
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(str(signal_id))
        tmp.replace(STATE_FILE)
    except Exception as exc:
        log.warning("could not persist state file %s: %s", STATE_FILE, exc)


# ---------------------------------------------------------------------------
# Kafka producer with exponential backoff
# ---------------------------------------------------------------------------


def build_producer() -> KafkaProducer:
    """Connect to Kafka with exponential backoff (capped at 60s)."""
    delay = 1.0
    max_delay = 60.0
    while not _shutdown:
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BROKER,
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                acks="all",
                retries=3,
                linger_ms=50,
                request_timeout_ms=15000,
            )
            log.info("connected to Kafka broker %s", KAFKA_BROKER)
            return producer
        except (NoBrokersAvailable, KafkaError, OSError) as exc:
            log.warning(
                "Kafka unreachable (%s); retrying in %.1fs", exc, delay
            )
            _sleep_interruptible(delay)
            delay = min(delay * 2, max_delay)
    raise RuntimeError("shutdown requested before Kafka connect")


def _sleep_interruptible(seconds: float) -> None:
    end = time.time() + seconds
    while not _shutdown and time.time() < end:
        time.sleep(min(0.5, max(0.0, end - time.time())))


# ---------------------------------------------------------------------------
# DB access (read-only)
# ---------------------------------------------------------------------------


SELECT_LATEST_SQL = """
    SELECT
        id,
        week,
        direction,
        confidence_tier,
        ensemble_return,
        skip_trade,
        hard_stop_pct,
        take_profit_pct,
        adjusted_leverage,
        created_at
    FROM forecast_h5_signals
    WHERE id > %s
    ORDER BY id ASC
    LIMIT 50
"""


def fetch_new_signals(last_id: int) -> list[dict[str, Any]]:
    if psycopg2 is None:
        log.error("psycopg2 not installed; cannot query DB")
        return []
    try:
        with psycopg2.connect(DATABASE_URL, connect_timeout=10) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(SELECT_LATEST_SQL, (last_id,))
                return [dict(row) for row in cur.fetchall()]
    except Exception as exc:
        log.warning("DB query failed: %s", exc)
        return []


def row_to_message(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize a DB row into the shared message contract."""
    created = row.get("created_at")
    if isinstance(created, datetime):
        ts = created.astimezone(_COT).isoformat()
    else:
        ts = datetime.now(tz=_COT).isoformat()

    return {
        "week": _as_str(row.get("week")),
        "direction": _as_str(row.get("direction")),
        "confidence": _as_float(row.get("confidence_tier")),
        "ensemble_return": _as_float(row.get("ensemble_return")),
        "skip_trade": bool(row.get("skip_trade") or False),
        "hard_stop_pct": _as_float(row.get("hard_stop_pct")),
        "take_profit_pct": _as_float(row.get("take_profit_pct")),
        "adjusted_leverage": _as_float(row.get("adjusted_leverage")),
        "timestamp": ts,
    }


def _as_str(value: Any) -> str:
    return "" if value is None else str(value)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------


def publish(producer: KafkaProducer, payload: dict[str, Any]) -> bool:
    try:
        future = producer.send(KAFKA_TOPIC, value=payload)
        meta = future.get(timeout=15)
        log.info(
            "published week=%s direction=%s to %s partition=%d offset=%d",
            payload.get("week"),
            payload.get("direction"),
            meta.topic,
            meta.partition,
            meta.offset,
        )
        return True
    except Exception as exc:
        log.warning("publish failed for week=%s: %s", payload.get("week"), exc)
        return False


# ---------------------------------------------------------------------------
# Demo mode
# ---------------------------------------------------------------------------


def demo(producer: KafkaProducer) -> int:
    """Publish 3 synthetic signals with increasing week numbers, then exit."""
    base_week = 17
    year = 2026
    directions = ["SHORT", "LONG", "SHORT"]
    successes = 0

    for i in range(3):
        payload = {
            "week": f"{year}-W{base_week + i:02d}",
            "direction": directions[i],
            "confidence": round(0.7 + 0.05 * i, 2),
            "ensemble_return": round(-0.012 + 0.004 * i, 4),
            "skip_trade": False,
            "hard_stop_pct": round(2.8 - 0.1 * i, 2),
            "take_profit_pct": round(1.4 - 0.05 * i, 2),
            "adjusted_leverage": round(1.5 - 0.1 * i, 2),
            "timestamp": datetime.now(tz=_COT).isoformat(),
        }
        if publish(producer, payload):
            successes += 1

    try:
        producer.flush(timeout=15)
    except Exception as exc:
        log.warning("flush failed at end of demo: %s", exc)

    log.info("demo complete: %d/3 published", successes)
    return 0 if successes == 3 else 1


# ---------------------------------------------------------------------------
# Main poll loop
# ---------------------------------------------------------------------------


def run_loop(producer: KafkaProducer) -> int:
    log.info(
        "entering poll loop: topic=%s interval=%ds db=%s",
        KAFKA_TOPIC,
        POLL_INTERVAL_SEC,
        _mask_url(DATABASE_URL),
    )
    last_id = load_last_published_id()
    log.info("starting from last_id=%d", last_id)

    while not _shutdown:
        rows = fetch_new_signals(last_id)
        if rows:
            log.info("found %d new signal(s) since id=%d", len(rows), last_id)
            for row in rows:
                if _shutdown:
                    break
                payload = row_to_message(row)
                if publish(producer, payload):
                    last_id = int(row["id"])
                    save_last_published_id(last_id)
        else:
            log.info("no new signals (last_id=%d)", last_id)

        _sleep_interruptible(POLL_INTERVAL_SEC)

    try:
        producer.flush(timeout=10)
    except Exception:
        pass
    log.info("producer exiting cleanly")
    return 0


def _mask_url(url: str) -> str:
    # Strip password for log hygiene: scheme://user:***@host/db
    try:
        if "@" in url and "://" in url:
            scheme, rest = url.split("://", 1)
            creds, host = rest.split("@", 1)
            user = creds.split(":", 1)[0]
            return f"{scheme}://{user}:***@{host}"
    except Exception:
        pass
    return url


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="USDCOP H5 Kafka producer")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="publish 3 synthetic signals and exit",
    )
    args = parser.parse_args()

    log.info(
        "kafka_bridge.producer starting: broker=%s topic=%s demo=%s",
        KAFKA_BROKER,
        KAFKA_TOPIC,
        args.demo,
    )

    try:
        producer = build_producer()
    except RuntimeError as exc:
        log.error("could not connect to Kafka: %s", exc)
        return 2

    try:
        if args.demo:
            return demo(producer)
        return run_loop(producer)
    finally:
        try:
            producer.close(timeout=5)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
