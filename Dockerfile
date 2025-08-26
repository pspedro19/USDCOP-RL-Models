# Multi-stage Dockerfile for USDCOP Trading System
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 trader && \
    mkdir -p /app /app/data /app/logs /app/models && \
    chown -R trader:trader /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/trader/.local

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=trader:trader . .

# Switch to non-root user
USER trader

# Add local bin to PATH
ENV PATH=/home/trader/.local/bin:$PATH
ENV PYTHONPATH=/app:$PYTHONPATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command
CMD ["python", "scripts/run_system.py", "live"]