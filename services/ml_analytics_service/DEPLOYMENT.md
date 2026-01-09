# ML Analytics Service - Deployment Guide

## Quick Start

### 1. Prerequisites

- Python 3.10+
- PostgreSQL database with `usdcop_trading` database
- Access to `dw.fact_rl_inference` table with inference data

### 2. Local Development

```bash
# Navigate to service directory
cd services/ml_analytics_service

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env

# Edit .env with your database credentials
# nano .env

# Run the service
python main.py
```

The service will start on http://localhost:8004

### 3. Test the Service

```bash
# Test database connection and services
python test_service.py

# Test API endpoints
python example_usage.py

# Or use curl
curl http://localhost:8004/health
```

### 4. Docker Deployment

```bash
# Build Docker image
docker build -t ml-analytics-service:latest .

# Run container
docker run -d \
  --name ml-analytics \
  -p 8004:8004 \
  -e POSTGRES_HOST=your-postgres-host \
  -e POSTGRES_PORT=5432 \
  -e POSTGRES_DB=usdcop_trading \
  -e POSTGRES_USER=admin \
  -e POSTGRES_PASSWORD=your-password \
  ml-analytics-service:latest

# View logs
docker logs -f ml-analytics

# Stop container
docker stop ml-analytics
```

## Docker Compose Integration

Add to your main `docker-compose.yml`:

```yaml
services:
  # ... other services ...

  ml-analytics-service:
    build:
      context: ./services/ml_analytics_service
      dockerfile: Dockerfile
    container_name: usdcop-ml-analytics
    ports:
      - "8004:8004"
    environment:
      - POSTGRES_HOST=usdcop-postgres-timescale
      - POSTGRES_PORT=5432
      - POSTGRES_DB=usdcop_trading
      - POSTGRES_USER=admin
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - DEBUG=false
    depends_on:
      - postgres-timescale
    networks:
      - usdcop-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8004/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

networks:
  usdcop-network:
    external: true
```

Start the service:

```bash
docker-compose up -d ml-analytics-service
```

## API Documentation

Once running, access the interactive API documentation:

- **Swagger UI**: http://localhost:8004/docs
- **ReDoc**: http://localhost:8004/redoc

## Common API Calls

### Get All Models Health Status

```bash
curl http://localhost:8004/api/health/models
```

### Get Metrics for a Specific Model

```bash
curl "http://localhost:8004/api/metrics/rolling?model_id=ppo_lstm_v3.2&window=24h"
```

### Check Drift Status

```bash
curl "http://localhost:8004/api/drift/status?model_id=ppo_lstm_v3.2"
```

### Get Prediction Accuracy

```bash
curl "http://localhost:8004/api/predictions/accuracy?model_id=ppo_lstm_v3.2&window=24h"
```

### Compare Model Performance

```bash
curl "http://localhost:8004/api/performance/comparison?window=24h"
```

## Monitoring

### Logs

```bash
# Docker logs
docker logs -f ml-analytics

# Local logs (stdout)
python main.py
```

### Health Check

```bash
# Service health
curl http://localhost:8004/health

# Expected response:
{
  "success": true,
  "service": "ML Analytics Service",
  "status": "healthy",
  "database": "connected",
  "timestamp": "2025-12-17T10:30:00Z"
}
```

## Troubleshooting

### Cannot Connect to Database

**Error**: `Database connection failed`

**Solution**:
1. Check database credentials in `.env`
2. Verify PostgreSQL is running
3. Check network connectivity
4. Verify `dw.fact_rl_inference` table exists

```bash
# Test database connection
psql -h localhost -U admin -d usdcop_trading -c "SELECT COUNT(*) FROM dw.fact_rl_inference;"
```

### No Data Found

**Error**: `No data found for this window`

**Solution**:
1. Verify inference data exists in `dw.fact_rl_inference`
2. Check data is within the requested time window
3. Verify model_id is correct

```sql
-- Check available models
SELECT DISTINCT model_id, COUNT(*) as predictions
FROM dw.fact_rl_inference
WHERE timestamp_utc >= NOW() - INTERVAL '7 days'
GROUP BY model_id;
```

### Port Already in Use

**Error**: `Address already in use: 8004`

**Solution**:
1. Change port in `.env`: `SERVICE_PORT=8005`
2. Or stop the conflicting service

```bash
# Find process using port 8004
lsof -i :8004  # macOS/Linux
netstat -ano | findstr :8004  # Windows

# Kill the process
kill -9 <PID>
```

### Import Errors

**Error**: `ModuleNotFoundError`

**Solution**:
1. Ensure virtual environment is activated
2. Reinstall dependencies

```bash
pip install -r requirements.txt
```

## Performance Tuning

### Connection Pool Size

Adjust in `database/postgres_client.py` or pass to `PostgresClient()`:

```python
db = PostgresClient(min_connections=5, max_connections=20)
```

### Query Limits

Default limits are set in service methods. Adjust if needed:

```python
# In services/metrics_calculator.py
inferences = self.db.get_inference_data(
    model_id=model_id,
    limit=10000  # Increase for more data
)
```

### Drift Detection Windows

Adjust in API calls:

```bash
# Shorter window for faster detection
curl "http://localhost:8004/api/drift/status?model_id=X&window_hours=6&baseline_days=3"

# Longer window for more stable metrics
curl "http://localhost:8004/api/drift/status?model_id=X&window_hours=48&baseline_days=14"
```

## Production Considerations

### 1. Security

- Use strong database passwords
- Enable SSL for database connections
- Configure CORS appropriately in `main.py`
- Add authentication middleware
- Use secrets management (AWS Secrets Manager, HashiCorp Vault)

### 2. Scaling

- Use multiple replicas with load balancer
- Implement Redis caching for frequently accessed metrics
- Consider read replicas for database queries

### 3. Monitoring

- Integrate with Prometheus for metrics collection
- Set up Grafana dashboards
- Configure alerting (PagerDuty, Slack)

### 4. Logging

- Use structured logging (JSON format)
- Send logs to centralized system (ELK, CloudWatch)
- Set appropriate log levels

### 5. Backup

- Regular database backups
- Document recovery procedures
- Test restore process

## Integration Examples

### Python Client

```python
import requests

class MLAnalyticsClient:
    def __init__(self, base_url="http://localhost:8004"):
        self.base_url = base_url

    def get_model_metrics(self, model_id, window='24h'):
        response = requests.get(
            f"{self.base_url}/api/metrics/rolling",
            params={'model_id': model_id, 'window': window}
        )
        return response.json()

    def get_drift_status(self, model_id):
        response = requests.get(
            f"{self.base_url}/api/drift/status",
            params={'model_id': model_id}
        )
        return response.json()

# Usage
client = MLAnalyticsClient()
metrics = client.get_model_metrics('ppo_lstm_v3.2')
print(metrics)
```

### JavaScript/React

```javascript
const API_BASE = 'http://localhost:8004';

async function getModelMetrics(modelId, window = '24h') {
  const response = await fetch(
    `${API_BASE}/api/metrics/rolling?model_id=${modelId}&window=${window}`
  );
  return await response.json();
}

// Usage
const metrics = await getModelMetrics('ppo_lstm_v3.2');
console.log(metrics);
```

## Support

For issues or questions:
- Check logs: `docker logs ml-analytics`
- Run test suite: `python test_service.py`
- Review API docs: http://localhost:8004/docs

## Version History

- **v1.0.0** (2025-12-17): Initial release
  - Rolling metrics calculation
  - Drift detection (data + concept)
  - Prediction tracking
  - Model health monitoring
  - Performance analysis
