# ================================================
# ML Analytics Service - Dockerfile
# ================================================
# Professional ML Analytics Backend for RL model monitoring
# Port: 8004
# ================================================

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 mlanalytics && \
    chown -R mlanalytics:mlanalytics /app

USER mlanalytics

# Expose port
EXPOSE 8004

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8004/health')" || exit 1

# Start service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]
