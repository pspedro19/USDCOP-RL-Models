#!/usr/bin/env bash
# Kafka bridge entrypoint.
#
# Dispatches to producer.py or consumer.py based on the MODE env var.
# Extra CLI args (e.g. --demo, --count 3 --timeout 30) are forwarded.

set -euo pipefail

MODE="${MODE:-producer}"

mkdir -p /data || true

case "${MODE}" in
  producer)
    echo "[entrypoint] starting producer (extra args: $*)"
    exec python -u /app/producer.py "$@"
    ;;
  consumer)
    echo "[entrypoint] starting consumer (extra args: $*)"
    exec python -u /app/consumer.py "$@"
    ;;
  *)
    echo "[entrypoint] ERROR: unknown MODE='${MODE}' (expected producer|consumer)" >&2
    exit 64
    ;;
esac
