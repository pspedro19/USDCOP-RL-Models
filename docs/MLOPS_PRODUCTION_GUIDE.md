# MLOps Production Guide

## Overview

This document describes the production MLOps system for USDCOP RL/ML trading models.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PRODUCTION INFERENCE FLOW                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Market Data ──► Feature Cache ──► ONNX Runtime ──► Risk Manager ──► API │
│       │              (Redis)         (Models)        (Circuit          │
│       │                                              Breaker)           │
│       ▼                                                                  │
│  Drift Monitor ────────────────────────────────────► Alerts             │
│   (Evidently)                                                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Inference Engine (`services/mlops/inference_engine.py`)

ONNX Runtime-based inference with:
- Sub-5ms latency (vs ~50ms with PyTorch)
- Ensemble support (weighted average, majority vote)
- Thread-safe predictions
- Model warming on startup

```python
from services.mlops import InferenceEngine

engine = InferenceEngine()
engine.load_models()

result = engine.predict(observation)  # Single model
result = engine.predict_ensemble(observation)  # Ensemble
```

### 2. Risk Manager (`services/mlops/risk_manager.py`)

Production risk management with:
- Circuit breaker pattern
- Daily loss limits
- Drawdown protection
- Position sizing recommendations

**Risk Limits (config/mlops.yaml):**
```yaml
risk_limits:
  max_daily_loss: -0.02      # -2% triggers circuit breaker
  max_drawdown: -0.05        # -5% triggers circuit breaker
  max_consecutive_losses: 5   # 5 consecutive losses stops trading
  min_confidence: 0.60        # 60% minimum to execute
  max_trades_per_day: 50      # Daily trade limit
```

### 3. Feature Cache (`services/mlops/feature_cache.py`)

Redis-based feature caching:
- TTL-based expiration (5 minutes default)
- Consistent feature ordering
- Batch retrieval support
- In-memory fallback

### 4. Drift Monitor (`services/mlops/drift_monitor.py`)

Data drift detection using Evidently AI:
- Kolmogorov-Smirnov test fallback
- Redis persistence
- HTML report generation
- Alerting integration

### 5. Configuration (`services/mlops/config.py`)

Centralized configuration from YAML:
```python
from services.mlops import get_config

config = get_config()
print(config.risk_limits.max_daily_loss)  # -0.02
print(config.trading_hours.start_hour)     # 8
```

## API Endpoints

The Inference API (`services/mlops/inference_api.py`) exposes:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with component status |
| `/v1/inference` | POST | Run model inference with risk checks |
| `/v1/inference/batch` | POST | Batch inference |
| `/v1/risk/summary` | GET | Current risk status |
| `/v1/risk/trade-result` | POST | Update with trade result |
| `/v1/risk/reset` | POST | Reset daily statistics |
| `/v1/metrics` | GET | Service metrics |
| `/v1/models` | GET | List loaded models |
| `/v1/cache/stats` | GET | Feature cache statistics |

### Example: Run Inference

```bash
curl -X POST http://localhost:8090/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "observation": [0.001, 0.002, ...],  # 45 features
    "use_ensemble": true,
    "enforce_risk_checks": true
  }'
```

Response:
```json
{
  "signal": "BUY",
  "confidence": 0.78,
  "approved": true,
  "risk_status": "APPROVED",
  "action_probs": {"SELL": 0.12, "HOLD": 0.10, "BUY": 0.78},
  "latency_ms": 3.45,
  "model_name": "ensemble",
  "timestamp": "2024-01-15T10:30:00",
  "risk_metrics": {
    "daily_pnl": 0.005,
    "trades_today": 12,
    "consecutive_losses": 0
  },
  "position_recommendation": {
    "position_size_percent": 0.08,
    "position_size_usd": 8000,
    "risk_level": "medium"
  }
}
```

## Deployment

### Docker Compose

The service is configured in `docker-compose.yml`:

```yaml
mlops-inference-api:
  build:
    context: ./services
    dockerfile: Dockerfile.api
    args:
      APP_FILE: mlops/inference_api.py
      PORT: 8090
  ports:
    - "8090:8090"
  environment:
    - MLOPS_CONFIG_PATH=/app/config/mlops.yaml
    - MODEL_STORAGE_PATH=/models/onnx
  volumes:
    - ./models:/models:ro
    - ./config:/app/config:ro
```

### Start Services

```bash
# Start all services
docker-compose up -d

# Start only MLOps inference
docker-compose up -d mlops-inference-api

# View logs
docker-compose logs -f mlops-inference-api
```

## Model Export

Export trained models to ONNX format:

```bash
# From project root
python scripts/mlops/export_model_onnx.py \
  --model-path models/trained/ppo_usdcop_v3.zip \
  --output-path models/onnx/ppo_usdcop_v3.onnx \
  --algorithm PPO \
  --observation-dim 45 \
  --verify
```

## Monitoring

### Airflow DAG

The `mlops_drift_monitor` DAG runs hourly during trading hours:
- Loads reference data (30 days)
- Compares against current data (1 hour)
- Detects feature drift
- Stores reports in Redis
- Sends alerts if drift > 15%

### Prometheus Metrics

Expose metrics at `/metrics` endpoint for Prometheus scraping.

### Grafana Dashboard

Import the dashboard from `grafana-dashboards/mlops-monitoring.json`.

## Trading Hours

The system operates during Colombian market hours:
- **Start**: 8:00 AM COT (America/Bogota)
- **End**: 12:55 PM COT
- **Days**: Monday - Friday

Outside these hours, signals default to HOLD.

## Risk Management Flow

```
1. Signal Generated (BUY/SELL)
   │
   ▼
2. Confidence Check
   ├─► < 60%: Signal → HOLD
   │
   ▼
3. Trading Hours Check
   ├─► Outside hours: Signal → HOLD
   │
   ▼
4. Circuit Breaker Check
   ├─► Active: Signal → HOLD
   │
   ▼
5. Daily Limits Check
   ├─► Exceeded: Signal → HOLD
   │
   ▼
6. Signal Approved ✓
```

## Testing

Run integration tests:

```bash
# All MLOps tests
pytest tests/integration/test_mlops_integration.py -v

# Specific test class
pytest tests/integration/test_mlops_integration.py::TestRiskManager -v

# With coverage
pytest tests/integration/test_mlops_integration.py --cov=services/mlops
```

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLOPS_CONFIG_PATH` | `config/mlops.yaml` | Path to config file |
| `MLOPS_ENVIRONMENT` | `production` | Environment mode |
| `REDIS_HOST` | `redis` | Redis hostname |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_PASSWORD` | - | Redis password |
| `MAX_DAILY_LOSS` | `-0.02` | Daily loss limit |
| `MAX_DRAWDOWN` | `-0.05` | Drawdown limit |
| `MIN_CONFIDENCE` | `0.60` | Minimum confidence |

### mlops.yaml Structure

```yaml
environment: production

models:
  - name: ppo_usdcop_v3
    algorithm: PPO
    onnx_path: models/onnx/ppo_usdcop_v3.onnx
    observation_dim: 45
    action_space_size: 3
    weight: 1.0
    enabled: true

risk_limits:
  max_daily_loss: -0.02
  max_drawdown: -0.05
  max_consecutive_losses: 5
  min_confidence: 0.60
  max_trades_per_day: 50
  max_position_size: 0.10

trading_hours:
  start_hour: 8
  end_hour: 12
  end_minute: 55
  timezone: America/Bogota
  trading_days: [0, 1, 2, 3, 4]

monitoring:
  enable_drift_detection: true
  drift_check_interval_minutes: 60
  drift_threshold: 0.15
```

## Troubleshooting

### Models Not Loading

1. Check model paths in `config/mlops.yaml`
2. Verify ONNX files exist in `models/onnx/`
3. Check observation dimension matches (45 features)

### High Latency

1. Warm up models after loading
2. Check Redis connection
3. Monitor system resources

### Circuit Breaker Active

1. Check `/v1/risk/summary` for current status
2. Review trading history for losses
3. Reset with `/v1/risk/reset` or wait for cooldown

### Drift Alerts

1. Check Airflow DAG logs
2. Review drift report in Redis: `redis-cli GET drift:latest`
3. Consider retraining models

## Files Reference

```
services/mlops/
├── __init__.py           # Module exports
├── config.py             # Configuration management
├── inference_engine.py   # ONNX Runtime inference
├── risk_manager.py       # Risk management
├── feature_cache.py      # Redis caching
├── drift_monitor.py      # Drift detection
└── inference_api.py      # FastAPI service

scripts/mlops/
└── export_model_onnx.py  # Model export script

config/
└── mlops.yaml            # Main configuration

airflow/dags/
└── mlops_drift_monitor.py # Drift monitoring DAG

models/onnx/
├── README.md             # Export instructions
├── ppo_usdcop_v3.onnx    # PPO model
├── sac_usdcop_v2.onnx    # SAC model
└── a2c_macro_v1.onnx     # A2C model
```
