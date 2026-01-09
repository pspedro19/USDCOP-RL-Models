# Multi-Model Trading Backend Documentation

**Version:** 1.0.0
**Last Updated:** 2025-12-26
**Maintainer:** USDCOP Trading Team

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Quick Start](#2-quick-start)
3. [API Reference](#3-api-reference)
4. [Database Schema](#4-database-schema)
5. [Adding a New Model](#5-adding-a-new-model)
6. [Feature Configuration](#6-feature-configuration)
7. [Redis Streams](#7-redis-streams)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Architecture Overview

### 1.1 System Diagram

```
                                    +------------------+
                                    |   Dashboard      |
                                    |   (Next.js)      |
                                    |   Port: 5000     |
                                    +--------+---------+
                                             |
                    +------------------------+------------------------+
                    |                        |                        |
           +--------v--------+     +---------v--------+     +---------v--------+
           | Trading API     |     | Multi-Model API  |     | Analytics API    |
           | (Real-time)     |     | (Strategies)     |     | (Statistics)     |
           | Port: 8000      |     | Port: 8006       |     | Port: 8001       |
           +--------+--------+     +---------+--------+     +---------+--------+
                    |                        |                        |
                    +------------------------+------------------------+
                                             |
                              +--------------v--------------+
                              |     PostgreSQL/TimescaleDB  |
                              |     (usdcop_trading)        |
                              |     Port: 5432              |
                              +-------------+---------------+
                                            |
                    +-----------------------+-----------------------+
                    |                       |                       |
           +--------v--------+    +---------v--------+    +---------v--------+
           | usdcop_m5_ohlcv |    | macro_indicators |    | dw.* Schema      |
           | (5min OHLCV)    |    | _daily (37 cols) |    | (Signals, Equity)|
           +-----------------+    +------------------+    +------------------+
```

### 1.2 Component Responsibilities

| Component | Port | Description |
|-----------|------|-------------|
| `trading-api` | 8000 | Real-time market data, WebSocket, candlesticks |
| `multi-model-api` | 8006 | Multi-strategy signals, performance comparison, equity curves |
| `analytics-api` | 8001 | Trading analytics, statistics, historical analysis |
| `postgres` | 5432 | TimescaleDB with market data and DW schema |
| `redis` | 6379 | Caching, pub/sub for real-time updates |
| `minio` | 9000/9001 | Object storage for model artifacts (L5/L6) |

### 1.3 Multi-Model Strategy Types

The system supports 5 trading strategies:

| Strategy Code | Type | Description |
|---------------|------|-------------|
| `RL_PPO` | RL | Proximal Policy Optimization reinforcement learning model |
| `ML_XGB` | ML | XGBoost gradient boosting classifier/regressor |
| `ML_LGBM` | ML | LightGBM gradient boosting model |
| `LLM_CLAUDE` | LLM | Claude-based natural language trading signals |
| `ENSEMBLE` | ENSEMBLE | Weighted combination of all strategies |

### 1.4 Data Flow

```
TwelveData API --> Airflow DAGs --> PostgreSQL --> APIs --> Dashboard
                      |                 |
                      v                 v
                   MinIO            Redis Cache
                 (L5/L6 artifacts)  (Real-time)
```

---

## 2. Quick Start

### 2.1 Start All Services

```bash
# Start the complete stack
docker-compose up -d

# Check all services are running
docker-compose ps
```

### 2.2 Verify Database Initialization

```bash
# Check essential tables exist
docker-compose exec postgres psql -U admin -d usdcop_trading -c "
  SELECT table_schema, table_name
  FROM information_schema.tables
  WHERE table_name IN ('usdcop_m5_ohlcv', 'macro_indicators_daily', 'users')
  OR table_schema = 'dw'
  ORDER BY table_schema, table_name;
"

# Check OHLCV data count
docker-compose exec postgres psql -U admin -d usdcop_trading -c "
  SELECT COUNT(*), MIN(time), MAX(time) FROM usdcop_m5_ohlcv;
"

# Check DW schema tables (for multi-model)
docker-compose exec postgres psql -U admin -d usdcop_trading -c "
  SELECT table_name FROM information_schema.tables
  WHERE table_schema = 'dw';
"
```

### 2.3 Verify APIs are Running

```bash
# Trading API (real-time data)
curl http://localhost:8000/api/health

# Multi-Model API (strategies)
curl http://localhost:8006/api/health

# Analytics API
curl http://localhost:8001/api/health

# Get available endpoints
curl http://localhost:8006/
```

### 2.4 Expected Health Response

```json
{
  "status": "healthy",
  "database": "connected",
  "timestamp": "2025-12-26T12:00:00.000Z"
}
```

---

## 3. API Reference

### 3.1 Multi-Model API Endpoints (Port 8006)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API root with endpoint listing |
| `/api/health` | GET | Health check |
| `/api/models/signals/latest` | GET | Latest signals from all strategies |
| `/api/models/performance/comparison` | GET | Performance metrics comparison |
| `/api/models/equity-curves` | GET | Historical equity curves |
| `/api/models/positions/current` | GET | Current open positions |
| `/api/models/pnl/summary` | GET | P&L summary by strategy |
| `/ws/trading-signals` | WS | Real-time signal updates |

### 3.2 Example: Get Latest Signals

```bash
curl http://localhost:8006/api/models/signals/latest
```

Response:
```json
{
  "timestamp": "2025-12-26T12:00:00.000Z",
  "market_price": 4250.50,
  "market_status": "open",
  "signals": [
    {
      "strategy_code": "RL_PPO",
      "strategy_name": "PPO Reinforcement Learning",
      "signal": "long",
      "side": "buy",
      "confidence": 0.85,
      "size": 0.25,
      "entry_price": 4250.00,
      "stop_loss": 4200.00,
      "take_profit": 4350.00,
      "risk_usd": 50.00,
      "reasoning": "Bullish momentum with positive macro indicators",
      "timestamp": "2025-12-26T11:55:00.000Z",
      "age_seconds": 300
    }
  ]
}
```

### 3.3 Example: Performance Comparison

```bash
curl "http://localhost:8006/api/models/performance/comparison?period=30d"
```

### 3.4 Example: Equity Curves

```bash
curl "http://localhost:8006/api/models/equity-curves?hours=24&resolution=5m"
```

### 3.5 WebSocket Connection

```javascript
const ws = new WebSocket('ws://localhost:8006/ws/trading-signals');

ws.onopen = () => {
  console.log('Connected to trading signals');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'signal_update') {
    console.log('New signal:', data);
  }
};

// Keep alive
setInterval(() => ws.send('ping'), 25000);
```

---

## 4. Database Schema

### 4.1 Core Tables

#### `usdcop_m5_ohlcv` (TimescaleDB Hypertable)
Primary OHLCV market data table.

| Column | Type | Description |
|--------|------|-------------|
| time | TIMESTAMPTZ | Bar timestamp (primary key) |
| symbol | TEXT | Trading pair (default: 'USD/COP') |
| open | DECIMAL(12,6) | Open price |
| high | DECIMAL(12,6) | High price |
| low | DECIMAL(12,6) | Low price |
| close | DECIMAL(12,6) | Close price |
| volume | BIGINT | Volume |
| source | TEXT | Data source |

#### `macro_indicators_daily`
37-column macroeconomic indicators table.

| Column | Type | Description |
|--------|------|-------------|
| fecha/date | DATE | Primary key |
| dxy | NUMERIC(10,4) | US Dollar Index |
| vix | NUMERIC(10,4) | CBOE VIX |
| embi | NUMERIC(10,4) | EMBI Colombia spread |
| brent | NUMERIC(10,4) | Brent crude oil |
| ... | ... | 33 more macro columns |

### 4.2 DW Schema Tables (Multi-Model)

#### `dw.dim_strategy`
Strategy dimension table.

| Column | Type | Description |
|--------|------|-------------|
| strategy_id | SERIAL | Primary key |
| strategy_code | VARCHAR(50) | Unique code (RL_PPO, ML_XGB, etc.) |
| strategy_name | VARCHAR(200) | Display name |
| strategy_type | VARCHAR(50) | Type (RL, ML, LLM, ENSEMBLE) |
| is_active | BOOLEAN | Active status |

#### `dw.fact_strategy_signals`
Trading signals fact table.

| Column | Type | Description |
|--------|------|-------------|
| signal_id | BIGSERIAL | Primary key |
| strategy_id | INT | FK to dim_strategy |
| timestamp_utc | TIMESTAMPTZ | Signal timestamp |
| signal | VARCHAR(20) | long/short/flat/close |
| side | VARCHAR(10) | buy/sell/hold |
| confidence | DECIMAL(5,4) | 0.0-1.0 |
| size | DECIMAL(5,4) | Position size 0.0-1.0 |
| entry_price | DECIMAL(12,6) | Entry price |
| stop_loss | DECIMAL(12,6) | Stop loss level |
| take_profit | DECIMAL(12,6) | Take profit level |
| risk_usd | DECIMAL(12,2) | Risk in USD |
| reasoning | TEXT | Signal reasoning |

#### `dw.fact_equity_curve`
Equity curve time series.

| Column | Type | Description |
|--------|------|-------------|
| equity_id | BIGSERIAL | Primary key |
| strategy_id | INT | FK to dim_strategy |
| timestamp_utc | TIMESTAMPTZ | Timestamp |
| equity_value | DECIMAL(15,2) | Equity value |
| return_since_start_pct | DECIMAL(10,4) | Total return % |
| current_drawdown_pct | DECIMAL(10,4) | Current drawdown % |

#### `dw.fact_strategy_performance`
Daily performance metrics.

| Column | Type | Description |
|--------|------|-------------|
| perf_id | BIGSERIAL | Primary key |
| strategy_id | INT | FK to dim_strategy |
| date_cot | DATE | Trading date |
| daily_return_pct | DECIMAL(10,4) | Daily return % |
| sharpe_ratio | DECIMAL(10,4) | Sharpe ratio |
| sortino_ratio | DECIMAL(10,4) | Sortino ratio |
| max_drawdown_pct | DECIMAL(10,4) | Max drawdown % |
| n_trades | INT | Number of trades |
| win_rate | DECIMAL(5,4) | Win rate 0.0-1.0 |

#### `dw.fact_strategy_positions`
Open and closed positions.

| Column | Type | Description |
|--------|------|-------------|
| position_id | BIGSERIAL | Primary key |
| strategy_id | INT | FK to dim_strategy |
| side | VARCHAR(10) | long/short |
| quantity | DECIMAL(15,6) | Position size |
| entry_price | DECIMAL(12,6) | Entry price |
| entry_time | TIMESTAMPTZ | Entry timestamp |
| exit_price | DECIMAL(12,6) | Exit price (if closed) |
| exit_time | TIMESTAMPTZ | Exit timestamp |
| status | VARCHAR(20) | open/closed |
| realized_pnl | DECIMAL(15,2) | Realized P&L |
| unrealized_pnl | DECIMAL(15,2) | Unrealized P&L |

---

## 5. Adding a New Model

### 5.1 Step 1: Insert Strategy Configuration

```sql
INSERT INTO dw.dim_strategy (
    strategy_code,
    strategy_name,
    strategy_type,
    description,
    initial_equity,
    is_active
) VALUES (
    'ML_NEW',
    'New ML Model',
    'ML',
    'Description of new model',
    10000.00,
    TRUE
);
```

### 5.2 Step 2: Create Model Artifact

Save your trained model to MinIO:

```python
from minio import Minio
import pickle

# Initialize MinIO client
client = Minio(
    "localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin123",
    secure=False
)

# Save model
model_bytes = pickle.dumps(your_model)
client.put_object(
    "99-common-trading-models",
    "ML_NEW/model_v1.pkl",
    io.BytesIO(model_bytes),
    len(model_bytes)
)
```

### 5.3 Step 3: Create Inference DAG

Create a new Airflow DAG in `airflow/dags/`:

```python
# l5_ml_new_inference.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'trading',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def run_inference():
    # Load model from MinIO
    # Generate predictions
    # Insert into dw.fact_strategy_signals
    pass

with DAG(
    'l5_ml_new_inference',
    default_args=default_args,
    description='ML_NEW model inference',
    schedule_interval='*/5 8-12 * * 1-5',  # Every 5 min during market hours
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    inference_task = PythonOperator(
        task_id='run_inference',
        python_callable=run_inference,
    )
```

### 5.4 Step 4: Verify Integration

```bash
# Check strategy exists
curl http://localhost:8006/api/models/signals/latest

# Should include ML_NEW in response
```

---

## 6. Feature Configuration

### 6.1 RL Model Features (13 dimensions)

The RL model uses 13 normalized features:

| Feature | Source | Normalization |
|---------|--------|---------------|
| log_ret_5m | OHLCV | clip[-0.05, 0.05] |
| log_ret_1h | OHLCV | clip[-0.05, 0.05] |
| log_ret_4h | OHLCV | clip[-0.05, 0.05] |
| rsi_9 | Python | 0-100 scaled |
| atr_pct | Python | percentage |
| adx_14 | Python | 0-100 scaled |
| dxy_z | SQL | z-score (mean=100.21, std=5.60) |
| dxy_change_1d | SQL | clip[-0.03, 0.03] |
| vix_z | SQL | z-score (mean=21.16, std=7.89) |
| embi_z | SQL | z-score (mean=322.01, std=62.68) |
| brent_change_1d | SQL | clip[-0.10, 0.10] |
| rate_spread | SQL | z-score (mean=7.03, std=1.41) |
| usdmxn_change_1d | Python | percentage change |

### 6.2 Feature Configuration File

See `/config/feature_config.json` for the Single Source of Truth (SSOT) configuration.

### 6.3 Adding New Features

1. Add column to source table or view
2. Update `inference_features_5m` materialized view
3. Update `feature_config.json`
4. Retrain model with new feature dimension

---

## 7. Redis Streams

### 7.1 Stream Names

| Stream | Purpose |
|--------|---------|
| `trading:signals` | Real-time trading signals |
| `trading:prices` | Price updates |
| `trading:positions` | Position changes |
| `trading:alerts` | System alerts |

### 7.2 Consumer Group Setup

```bash
# Create consumer group
docker-compose exec redis redis-cli -a ${REDIS_PASSWORD} \
  XGROUP CREATE trading:signals dashboard-group $ MKSTREAM

# Read from stream
docker-compose exec redis redis-cli -a ${REDIS_PASSWORD} \
  XREAD COUNT 10 STREAMS trading:signals 0
```

### 7.3 Publishing to Stream

```python
import redis

r = redis.Redis(host='localhost', port=6379, password='your_password')

# Publish signal
r.xadd('trading:signals', {
    'strategy': 'RL_PPO',
    'signal': 'long',
    'confidence': '0.85',
    'timestamp': '2025-12-26T12:00:00Z'
})
```

### 7.4 Monitoring Streams

```bash
# Stream length
docker-compose exec redis redis-cli -a ${REDIS_PASSWORD} XLEN trading:signals

# Stream info
docker-compose exec redis redis-cli -a ${REDIS_PASSWORD} XINFO STREAM trading:signals
```

---

## 8. Troubleshooting

### 8.1 Common Issues

#### Database Connection Failed

```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Check logs
docker-compose logs postgres

# Test connection
docker-compose exec postgres pg_isready -U admin -d usdcop_trading
```

#### API Returns 500 Error

```bash
# Check API logs
docker-compose logs multi-model-api

# Common issues:
# 1. DW schema tables missing - run init scripts
# 2. Database connection string wrong - check env vars
# 3. Required modules not installed - check Dockerfile
```

#### No Signals Returned

```sql
-- Check if dim_strategy has active strategies
SELECT * FROM dw.dim_strategy WHERE is_active = TRUE;

-- Check if signals exist
SELECT COUNT(*) FROM dw.fact_strategy_signals
WHERE timestamp_utc > NOW() - INTERVAL '1 day';
```

#### WebSocket Connection Drops

```javascript
// Implement reconnection logic
let ws;
function connect() {
  ws = new WebSocket('ws://localhost:8006/ws/trading-signals');
  ws.onclose = () => setTimeout(connect, 5000);
}
connect();
```

### 8.2 Log Locations

| Service | Log Location |
|---------|--------------|
| Trading API | `docker-compose logs trading-api` |
| Multi-Model API | `docker-compose logs multi-model-api` |
| PostgreSQL | `docker-compose logs postgres` |
| Redis | `docker-compose logs redis` |
| Airflow | `./airflow/logs/` |

### 8.3 Health Checks

```bash
# Full system health check
./scripts/validate_backend_setup.py

# Individual checks
curl http://localhost:8000/api/health  # Trading API
curl http://localhost:8006/api/health  # Multi-Model API
curl http://localhost:8001/api/health  # Analytics API
```

### 8.4 Database Maintenance

```bash
# Vacuum and analyze
docker-compose exec postgres psql -U admin -d usdcop_trading -c "
  VACUUM ANALYZE usdcop_m5_ohlcv;
  VACUUM ANALYZE macro_indicators_daily;
"

# Check hypertable status
docker-compose exec postgres psql -U admin -d usdcop_trading -c "
  SELECT * FROM timescaledb_information.hypertables;
"
```

---

## Related Documentation

- [API Endpoints Reference](./API_ENDPOINTS_MULTIMODEL.md)
- [Database Schema V19](./DATABASE_SCHEMA_V19.md)
- [Architecture Overview V3](./ARQUITECTURA_INTEGRAL_V3.md)
- [Feature Configuration](../config/feature_config.json)

---

**Need Help?**
Check the troubleshooting section or run the validation script:
```bash
python scripts/validate_backend_setup.py
```
