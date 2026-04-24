# kafka_bridge

Kafka producer + consumer bridge for USDCOP H5 trading signals.
Satisfies the "Kafka (Redpanda)" technology requirement of the MLOps course project.

The **producer** polls `forecast_h5_signals` (PostgreSQL) every 60 seconds and
publishes new rows to the Kafka topic `signals.h5`. A cursor file at
`/data/published.txt` stores the last published signal `id` so restarts do not
re-publish.

The **consumer** subscribes to the same topic under group
`signalbridge-consumer`, prints each message to stdout, and appends to
`/data/consumed.log`.

## Shared contract

| Key | Value |
| --- | --- |
| Broker env | `KAFKA_BROKER` (default `redpanda:9092`, host `localhost:19092`) |
| Topic | `signals.h5` |
| Consumer group | `signalbridge-consumer` |
| DB env | `DATABASE_URL` (fallback `postgresql://admin:admin123@postgres:5432/usdcop_trading`) |
| State dir | `/data` (cursor + consumed log) |

Message payload:

```json
{
  "week": "2026-W17",
  "direction": "SHORT",
  "confidence": 0.85,
  "ensemble_return": -0.012,
  "skip_trade": false,
  "hard_stop_pct": 2.8,
  "take_profit_pct": 1.4,
  "adjusted_leverage": 1.5,
  "timestamp": "2026-04-23T14:00:00-05:00"
}
```

The producer is **read-only** on `forecast_h5_signals` and never writes to the DB.

## Files

```
services/kafka_bridge/
├── Dockerfile           python:3.11-slim + tini + kafka-python + psycopg2-binary
├── entrypoint.sh        dispatches on MODE=producer|consumer
├── producer.py          DB poll → Kafka publish, with --demo
├── consumer.py          Kafka subscribe → stdout + /data/consumed.log, with --count/--timeout
├── requirements.txt     kafka-python>=2.0, psycopg2-binary, python-dateutil
└── README.md
```

## Build

```bash
docker build -t usdcop-kafka-bridge ./services/kafka_bridge
```

## Run (standalone, expects Redpanda reachable)

Producer — continuous poll of the DB, publishes new signals every 60s:

```bash
docker run --rm \
  --network usdcop-trading-network \
  -e MODE=producer \
  -e KAFKA_BROKER=redpanda:9092 \
  -e DATABASE_URL=postgresql://admin:admin123@postgres:5432/usdcop_trading \
  -v kafka_bridge_state:/data \
  usdcop-kafka-bridge
```

Producer — demo (publishes 3 synthetic signals and exits):

```bash
docker run --rm \
  --network usdcop-trading-network \
  -e MODE=producer \
  -e KAFKA_BROKER=redpanda:9092 \
  usdcop-kafka-bridge --demo
```

Consumer — continuous:

```bash
docker run --rm \
  --network usdcop-trading-network \
  -e MODE=consumer \
  -e KAFKA_BROKER=redpanda:9092 \
  -v kafka_bridge_state:/data \
  usdcop-kafka-bridge
```

Consumer — bounded (for demos / CI):

```bash
docker run --rm \
  --network usdcop-trading-network \
  -e MODE=consumer \
  -e KAFKA_BROKER=redpanda:9092 \
  usdcop-kafka-bridge --count 3 --timeout 30
```

## Run from the host (no Docker)

With Redpanda exposing `localhost:19092`:

```bash
cd services/kafka_bridge
python -m pip install -r requirements.txt

export KAFKA_BROKER=localhost:19092
export STATE_DIR=/tmp/kafka_bridge   # local sandbox instead of /data

# Terminal 1: consumer (bounded)
python consumer.py --count 3 --timeout 30

# Terminal 2: producer demo
python producer.py --demo
```

## Robustness

| Scenario | Producer | Consumer |
| --- | --- | --- |
| Broker down at startup | Exponential backoff to 60s, never crashes | 30 attempts x 10s, then exits with error |
| DB unreachable | Log warning, skip this tick, retry next poll | N/A |
| Bad message payload | Logged, message skipped | Logged, message skipped |
| SIGTERM / SIGINT | Flushes producer, exits 0 | Closes consumer, exits 0 |

## CLI reference

```
producer.py [--demo]
  --demo           publish 3 synthetic signals and exit

consumer.py [--count N] [--timeout SECONDS]
  --count N        stop after receiving N messages
  --timeout SEC    stop after SEC idle/elapsed seconds (pairs with --count)
```
