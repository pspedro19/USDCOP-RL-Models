# ML Analytics Service

Professional ML Analytics Backend for monitoring RL model performance in the USD/COP trading system.

## Overview

The ML Analytics Service provides comprehensive monitoring, analysis, and health tracking for reinforcement learning models in production. It connects to the `usdcop_trading` PostgreSQL database to analyze inference data, detect drift, track predictions, and provide actionable insights.

## Features

### 1. Rolling Metrics Calculation
- Calculate performance metrics over configurable time windows (1h, 24h, 7d, 30d)
- Metrics include:
  - Accuracy, Precision, Recall, F1 Score
  - MSE, MAE
  - Sharpe Ratio (annualized)
  - Profit Factor
  - Win Rate
  - Maximum Drawdown
  - Average Inference Latency

### 2. Drift Detection
- **Data Drift**: Detect feature distribution shifts using Kolmogorov-Smirnov test
- **Concept Drift**: Track changes in prediction accuracy
- Per-feature drift analysis
- Configurable warning and critical thresholds
- Status: `healthy` | `warning` | `critical`

### 3. Prediction Tracking
- Track predictions vs actual outcomes
- Prediction accuracy by action type (LONG, SHORT, HOLD)
- Historical prediction analysis
- Confusion matrix generation
- Prediction comparison analysis

### 4. Model Health Monitoring
- Real-time health status for all models
- Health scores (0-100) based on:
  - Win rate performance
  - Inference latency
  - Prediction recency
- Issue detection and alerts

### 5. Performance Analysis
- Performance trends over time
- Multi-model comparison
- Hourly/daily aggregations
- Trend detection (improving, declining, stable)

## Architecture

```
ml_analytics_service/
├── main.py                    # FastAPI application (port 8004)
├── config.py                  # Service configuration
├── models/                    # Pydantic schemas
│   ├── metrics_schema.py
│   └── prediction_schema.py
├── services/                  # Business logic
│   ├── metrics_calculator.py
│   ├── drift_detector.py
│   ├── prediction_tracker.py
│   └── performance_analyzer.py
├── api/                       # API routes
│   └── routes.py
├── database/                  # Database client
│   └── postgres_client.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## API Endpoints

### Metrics Endpoints

#### GET `/api/metrics/summary`
Get overall metrics summary for all models.

**Response:**
```json
{
  "success": true,
  "data": {
    "timestamp": "2025-12-17T10:30:00Z",
    "total_models": 5,
    "active_models": 5,
    "models": [...],
    "aggregate": {
      "accuracy": 0.72,
      "sharpe_ratio": 1.85,
      "win_rate": 0.68
    }
  }
}
```

#### GET `/api/metrics/rolling?model_id=ppo_lstm_v3.2&window=24h`
Get rolling window metrics for a specific model.

**Parameters:**
- `model_id` (required): Model identifier
- `window` (optional): Time window (`1h`, `24h`, `7d`, `30d`)

**Response:**
```json
{
  "success": true,
  "data": {
    "model_id": "ppo_lstm_v3.2",
    "window": "24h",
    "data_points": 1250,
    "metrics": {
      "accuracy": 0.72,
      "precision": 0.68,
      "recall": 0.75,
      "f1_score": 0.71,
      "mse": 0.0012,
      "mae": 0.025,
      "sharpe_ratio": 1.85,
      "profit_factor": 2.1,
      "win_rate": 0.68,
      "max_drawdown": 0.08,
      "avg_latency_ms": 45
    }
  }
}
```

### Prediction Endpoints

#### GET `/api/predictions/accuracy?model_id=ppo_lstm_v3.2&window=24h`
Get prediction accuracy metrics.

**Response:**
```json
{
  "success": true,
  "data": {
    "model_id": "ppo_lstm_v3.2",
    "total_predictions": 1250,
    "correct_predictions": 900,
    "accuracy": 0.72,
    "by_action": [
      {
        "action": "LONG",
        "count": 450,
        "avg_confidence": 0.75,
        "success_rate": 0.73
      }
    ]
  }
}
```

#### GET `/api/predictions/history?model_id=ppo_lstm_v3.2&page=1&page_size=100`
Get prediction history with outcomes.

#### GET `/api/predictions/comparison?model_id=ppo_lstm_v3.2&limit=100`
Compare predictions vs actual outcomes.

### Drift Detection Endpoints

#### GET `/api/drift/status?model_id=ppo_lstm_v3.2`
Get drift detection status.

**Response:**
```json
{
  "success": true,
  "data": {
    "model_id": "ppo_lstm_v3.2",
    "data_drift_score": 0.12,
    "concept_drift_score": 0.08,
    "status": "healthy",
    "features_drifted": ["rsi_14", "volume"],
    "last_check": "2025-12-17T10:30:00Z"
  }
}
```

#### GET `/api/drift/features?model_id=ppo_lstm_v3.2`
Get drift detection results grouped by feature.

### Health & Performance Endpoints

#### GET `/api/health/models`
Get health status for all models.

**Response:**
```json
{
  "success": true,
  "data": {
    "total_models": 5,
    "healthy_models": 4,
    "warning_models": 1,
    "critical_models": 0,
    "models": [
      {
        "model_id": "ppo_lstm_v3.2",
        "status": "healthy",
        "health_score": 85,
        "win_rate": 0.68,
        "avg_latency_ms": 45,
        "issues": []
      }
    ]
  }
}
```

#### GET `/api/health/model/{model_id}`
Get health status for a specific model.

#### GET `/api/performance/trends/{model_id}?days=7`
Get performance trends over time.

#### GET `/api/performance/comparison?window=24h`
Compare performance across all models.

#### GET `/health`
Service health check.

## Installation & Usage

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=usdcop_trading
export POSTGRES_USER=admin
export POSTGRES_PASSWORD=admin123
export SERVICE_PORT=8004
```

3. Run the service:
```bash
python main.py
```

4. Access the API documentation:
- Swagger UI: http://localhost:8004/docs
- ReDoc: http://localhost:8004/redoc

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t ml-analytics-service:latest .
```

2. Run the container:
```bash
docker run -d \
  --name ml-analytics \
  -p 8004:8004 \
  -e POSTGRES_HOST=usdcop-postgres-timescale \
  -e POSTGRES_PORT=5432 \
  -e POSTGRES_DB=usdcop_trading \
  -e POSTGRES_USER=admin \
  -e POSTGRES_PASSWORD=admin123 \
  ml-analytics-service:latest
```

### Docker Compose

Add to your `docker-compose.yml`:

```yaml
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
    - POSTGRES_PASSWORD=admin123
  depends_on:
    - postgres-timescale
  networks:
    - usdcop-network
  restart: unless-stopped
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVICE_HOST` | `0.0.0.0` | Service host |
| `SERVICE_PORT` | `8004` | Service port |
| `POSTGRES_HOST` | `usdcop-postgres-timescale` | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `usdcop_trading` | Database name |
| `POSTGRES_USER` | `admin` | Database user |
| `POSTGRES_PASSWORD` | `admin123` | Database password |
| `DEBUG` | `false` | Debug mode |
| `RELOAD` | `false` | Auto-reload on code changes |

### Drift Detection Thresholds

Configure in `config.py`:
- `drift_warning_threshold`: 0.15 (default)
- `drift_critical_threshold`: 0.30 (default)

## Database Schema

The service reads from the following tables:

### `dw.fact_rl_inference`
Main inference log table containing:
- Model predictions (action, confidence)
- Feature values (13 features)
- Performance metrics (reward, position changes)
- Metadata (latency, timestamps)

### `usdcop_m5_ohlcv`
OHLCV market data for context and actual returns.

## Metrics Calculation Details

### Sharpe Ratio
```python
# Annualized Sharpe Ratio
# Assuming 252 trading days * 59 bars/day = 14,868 bars/year
sharpe = (avg_return / std_return) * sqrt(14868)
```

### Profit Factor
```python
profit_factor = gross_profit / gross_loss
```

### Maximum Drawdown
```python
cumulative_returns = cumsum(rewards)
running_max = maximum.accumulate(cumulative_returns)
drawdown = running_max - cumulative_returns
max_drawdown = max(drawdown)
```

### Health Score (0-100)
```python
win_rate_score = win_rate * 80  # 0-40 points
latency_score = 30 if latency <= 50ms else 0  # 0-30 points
recency_score = 30 if <10min else 0  # 0-30 points
health_score = win_rate_score + latency_score + recency_score
```

## Monitoring & Logging

The service logs all operations with timestamps and severity levels:
- `INFO`: Normal operations
- `WARNING`: Degraded performance or drift warnings
- `ERROR`: Operation failures

## Error Handling

All endpoints return standardized error responses:
```json
{
  "success": false,
  "error": "Error description",
  "detail": "Detailed error message",
  "timestamp": "2025-12-17T10:30:00Z"
}
```

## Performance Considerations

- Connection pooling: 2-10 connections
- Query limits: Default 1000-10000 rows
- Pagination: 100 results per page (max 1000)
- Caching: Consider adding Redis for frequently accessed metrics

## Future Enhancements

- [ ] Redis caching for hot metrics
- [ ] Anomaly detection using isolation forests
- [ ] Alert system (email/Slack notifications)
- [ ] Historical metric storage and comparison
- [ ] A/B testing framework
- [ ] Model versioning and rollback support
- [ ] Custom metric definitions via API

## Contributing

Created and maintained by Pedro @ Lean Tech Solutions.

## License

Internal use only - USD/COP RL Trading System.
