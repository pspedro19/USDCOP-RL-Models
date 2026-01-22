# ================================================
# Backtest Inference API Dockerfile
# Provides /v1/backtest endpoints for trade generation
# ================================================

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# Install system dependencies including TA-Lib
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    wget \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib C library
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Copy requirements
COPY services/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for SB3
RUN pip install --no-cache-dir stable-baselines3==2.3.2 shimmy>=2.0

# Copy source code
COPY src/ ./src/
COPY services/inference_api/ ./services/inference_api/
COPY services/common/ ./services/common/
COPY services/demo_mode/ ./services/demo_mode/
COPY services/shared/ ./services/shared/
COPY config/ ./config/

# Copy migration scripts and init-scripts
COPY scripts/ ./scripts/
COPY init-scripts/ ./init-scripts/

# Copy and setup entrypoint
COPY services/inference_api/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Health check now includes schema validation
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Use entrypoint for migration + startup
ENTRYPOINT ["/entrypoint.sh"]
