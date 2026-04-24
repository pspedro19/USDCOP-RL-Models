# USDCOP H5 gRPC Prediction Service
#
# Builds a self-contained image that:
#   1. Installs grpcio + tooling + sklearn.
#   2. Generates predictor_pb2*.py from the proto at build time.
#   3. Runs server.py, binding 0.0.0.0:50051.
#
# Build:
#   docker build -t usdcop-grpc-predictor:latest services/grpc_predictor
#
# Run (bind H5 models if available; harmless if missing -- stub kicks in):
#   docker run --rm -p 50051:50051 \
#       -v "$PWD/outputs:/app/outputs:ro" \
#       usdcop-grpc-predictor:latest

FROM python:3.11-slim

# Don't write .pyc, keep stdout unbuffered for clean container logs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRPC_PORT=50051 \
    H5_MODEL_DIR=/app/outputs/forecasting/h5_weekly_models/latest

WORKDIR /app

# ---- dependencies ---------------------------------------------------------
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# ---- source --------------------------------------------------------------
COPY proto/              /app/proto/
COPY server.py           /app/server.py
COPY client_example.py   /app/client_example.py
COPY generate_pb2.sh     /app/generate_pb2.sh

# ---- generate pb2 at build time ------------------------------------------
RUN chmod +x /app/generate_pb2.sh && \
    /app/generate_pb2.sh && \
    python -c "import predictor_pb2, predictor_pb2_grpc; print('pb2 OK')"

# ---- runtime -------------------------------------------------------------
EXPOSE 50051

# Best-effort healthcheck via the HealthCheck RPC. Falls back to a simple
# TCP probe if grpc_health_probe is unavailable (it's not installed here).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import socket,sys; s=socket.socket(); s.settimeout(2); \
        s.connect(('127.0.0.1', 50051)); s.close()" || exit 1

CMD ["python", "/app/server.py"]
