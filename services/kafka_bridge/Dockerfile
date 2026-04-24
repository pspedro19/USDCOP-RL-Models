FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps for psycopg2-binary wheels are already included, but we keep
# build-essential available in case of source fallback.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libpq5 \
        ca-certificates \
        tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY producer.py /app/producer.py
COPY consumer.py /app/consumer.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Persistent state (published.txt cursor, consumed.log sink)
RUN mkdir -p /data
VOLUME ["/data"]

ENV MODE=producer \
    KAFKA_BROKER=redpanda:9092 \
    KAFKA_TOPIC=signals.h5 \
    CONSUMER_GROUP=signalbridge-consumer \
    STATE_DIR=/data

# tini handles PID 1 signal forwarding so SIGTERM reaches the Python process.
ENTRYPOINT ["/usr/bin/tini", "--", "/app/entrypoint.sh"]
CMD []
