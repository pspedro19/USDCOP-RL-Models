# USDCOP Trading System - Architecture Documentation

**Version:** 2.0.0
**Date:** October 22, 2025
**Status:** Production Ready

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Principles](#architecture-principles)
3. [Component Architecture](#component-architecture)
4. [Data Flow](#data-flow)
5. [Service Layer](#service-layer)
6. [Storage Layer](#storage-layer)
7. [Real-Time Processing](#real-time-processing)
8. [Design Decisions](#design-decisions)
9. [Comparison: Old vs New Architecture](#comparison-old-vs-new-architecture)

---

## System Overview

The USDCOP Trading System is a production-ready reinforcement learning platform for trading the USD/COP currency pair. It implements a complete 7-layer data pipeline (L0-L6) with real-time data collection, processing, and serving capabilities.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     USDCOP Trading System v2.0                       │
│                  Real-Time RL Trading Platform                       │
└─────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────┐
│                          PRESENTATION LAYER                            │
├───────────────────────────────────────────────────────────────────────┤
│  Next.js Dashboard (Port 5000)                                        │
│  - Trading Terminal                                                    │
│  - ML Analytics                                                        │
│  - Pipeline Diagnostics                                                │
│  - Real-time Charts (WebSocket)                                        │
└─────────────────────────────────┬─────────────────────────────────────┘
                                  │
┌─────────────────────────────────┴─────────────────────────────────────┐
│                          API/SERVICE LAYER                             │
├────────────────┬───────────────┬───────────────┬─────────────────────┤
│  Trading API   │ Analytics API │ Pipeline API  │ Compliance API      │
│  (Port 8000)   │ (Port 8001)   │ (Port 8002)   │ (Port 8003)         │
│  - Market Data │ - RL Metrics  │ - L0-L6 Data  │ - Audit Trails      │
│  - Positions   │ - Performance │ - Health      │ - Regulations       │
│  - Orders      │ - Backtests   │ - Quality     │ - Compliance        │
└────────────────┴───────────────┴───────────────┴─────────────────────┘
                                  │
┌─────────────────────────────────┴─────────────────────────────────────┐
│                      ORCHESTRATION LAYER                               │
├───────────────────────────────────────────────────────────────────────┤
│  RT Orchestrator (Port 8085)          │  Airflow (Port 8080)          │
│  - Market Hours Detection             │  - L0-L6 DAG Scheduling       │
│  - L0 Pipeline Dependency Manager     │  - Gap Detection/Filling      │
│  - Real-time Data Collection          │  - MinIO Manifest Management  │
│  - WebSocket Broadcasting             │  - 16 API Key Pool Management │
│  - Redis Pub/Sub                      │  - Intelligent Retry Logic    │
└───────────────────────────────────────┴───────────────────────────────┘
                                  │
┌─────────────────────────────────┴─────────────────────────────────────┐
│                         DATA/STORAGE LAYER                             │
├────────────────┬──────────────────┬──────────────────┬───────────────┤
│  PostgreSQL    │  TimescaleDB     │  MinIO (S3)      │  Redis        │
│  (Port 5432)   │  (Hypertables)   │  (Port 9000)     │  (Port 6379)  │
│  - OLTP Data   │  - Time Series   │  - Cold Storage  │  - Cache      │
│  - Metadata    │  - OHLCV Data    │  - Parquet Files │  - Pub/Sub    │
│  - Pipeline    │  - Auto-chunk    │  - All 7 Layers  │  - Session    │
│    Status      │    by Time       │  - Manifests     │  - RT Queue   │
└────────────────┴──────────────────┴──────────────────┴───────────────┘
                                  │
┌─────────────────────────────────┴─────────────────────────────────────┐
│                        EXTERNAL SERVICES                               │
├───────────────────────────────────────────────────────────────────────┤
│  TwelveData API (Market Data Provider)                                │
│  - 16 API Keys (2 groups of 8)                                        │
│  - 5-minute OHLCV data                                                │
│  - Colombia market hours (8 AM - 12:55 PM COT)                        │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Architecture Principles

### 1. Separation of Concerns

**Data Ingestion (L0) vs Real-Time (RT Orchestrator)**
- **L0 Pipeline**: Batch processing, historical gap filling, MinIO archival
- **RT Orchestrator**: Live streaming, low-latency updates, WebSocket broadcasting

**Why Separate?**
- Different SLAs: L0 = eventual consistency, RT = real-time
- Different failure modes: L0 can retry gaps, RT needs immediate fallback
- Different clients: L0 feeds ML training, RT feeds live dashboard

### 2. Dependency Management

**RT Orchestrator waits for L0 Pipeline**
```
08:00 AM COT: Market Opens
    ├─> L0 Pipeline starts (Airflow DAG)
    │   ├─> Fetches 0-5 min data
    │   ├─> Validates completeness
    │   ├─> Writes to PostgreSQL + MinIO
    │   └─> Updates pipeline_status table
    │
    ├─> RT Orchestrator polls L0 status
    │   ├─> Checks pipeline_status every 60s
    │   ├─> Max wait: 30 minutes
    │   └─> On success: starts RT collection
    │
    └─> WebSocket clients connect
        └─> Receive live updates
```

**Rationale**: Prevents duplicate data writes and ensures data consistency.

### 3. Storage Strategy

**Hot vs Cold Storage**

| Data Type | Storage | Retention | Query Speed | Purpose |
|-----------|---------|-----------|-------------|---------|
| **Latest 6 months** | PostgreSQL | Active | <100ms | Dashboard serving |
| **All history** | MinIO | Infinite | 1-5s | Training, backtest, audit |
| **Live cache** | Redis | 1 hour | <10ms | WebSocket broadcast |

**Why Dual Storage?**
- PostgreSQL: Fast queries for dashboard (indexed, time-series optimized)
- MinIO: Cheap archival, replay capability, regulatory compliance
- Redis: Real-time distribution, prevents DB overload

### 4. Scalability

**Horizontal Scaling Points**
1. **API Layer**: Stateless FastAPI services (can run N replicas)
2. **WebSocket**: Redis pub/sub allows multi-instance WebSocket servers
3. **Airflow**: CeleryExecutor for distributed task execution (currently LocalExecutor)
4. **MinIO**: Distributed mode for petabyte-scale storage

**Vertical Scaling Points**
1. **PostgreSQL**: TimescaleDB compression, automatic chunking
2. **Redis**: Redis Cluster for multi-GB datasets
3. **Airflow**: Worker concurrency tuning

### 5. Observability

**Four Pillars of Observability**

1. **Logs**: Structured JSON logs from all services
   - Aggregated via Docker logs
   - Searchable with `docker logs <container> | grep "ERROR"`

2. **Metrics**: Prometheus + Grafana
   - API latency (p50, p95, p99)
   - Pipeline success rate
   - WebSocket connection count
   - Database query performance

3. **Traces**: Request tracing (future enhancement)
   - Would use OpenTelemetry
   - Track request flow across services

4. **Health Checks**: Built into every service
   - HTTP `/health` endpoints
   - Docker healthchecks
   - Liveness probes for Kubernetes (future)

---

## Component Architecture

### 1. Real-Time Orchestrator

**File**: `services/usdcop_realtime_orchestrator.py`
**Port**: 8085
**Purpose**: Coordinate real-time data collection with pipeline dependencies

```python
┌─────────────────────────────────────────────────────────────┐
│            RT Orchestrator Components                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────────────────────────────┐                 │
│  │  Market Hours Manager                 │                 │
│  │  - COT timezone (America/Bogota)      │                 │
│  │  - 8:00 AM - 12:55 PM check           │                 │
│  │  - Monday-Friday only                 │                 │
│  └───────────────┬───────────────────────┘                 │
│                  │                                          │
│  ┌───────────────▼───────────────────────┐                 │
│  │  Pipeline Dependency Manager          │                 │
│  │  - Polls pipeline_status table        │                 │
│  │  - Waits max 30 min for L0            │                 │
│  │  - Fallback: checks historical data   │                 │
│  └───────────────┬───────────────────────┘                 │
│                  │                                          │
│  ┌───────────────▼───────────────────────┐                 │
│  │  Real-Time Data Collector             │                 │
│  │  - TwelveData API client              │                 │
│  │  - 5-second polling interval          │                 │
│  │  - Decimal precision handling         │                 │
│  └───────────────┬───────────────────────┘                 │
│                  │                                          │
│  ┌───────────────▼───────────────────────┐                 │
│  │  WebSocket Broadcaster                │                 │
│  │  - Redis pub/sub for multi-client     │                 │
│  │  - JSON message formatting            │                 │
│  │  - Automatic reconnection             │                 │
│  └───────────────┬───────────────────────┘                 │
│                  │                                          │
│  ┌───────────────▼───────────────────────┐                 │
│  │  Database Writer                      │                 │
│  │  - Asyncpg connection pool            │                 │
│  │  - Upsert strategy (avoid duplicates) │                 │
│  │  - Transaction safety                 │                 │
│  └───────────────────────────────────────┘                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key Features**:
- **Non-blocking**: Uses asyncio for concurrent operations
- **Resilient**: Retries on API failures with exponential backoff
- **Smart**: Only runs during market hours, saves API credits
- **Coordinated**: Respects L0 pipeline completion

### 2. Airflow Data Pipeline

**Location**: `airflow/dags/`
**Purpose**: 7-layer medallion architecture for data transformation

```
L0 (Raw Ingestion)
    ├─> DAG: usdcop_m5__01_l0_intelligent_acquire.py
    ├─> Input: TwelveData API
    ├─> Output: PostgreSQL + MinIO (00-raw-usdcop-marketdata)
    ├─> Features:
    │   ├─> Intelligent gap detection
    │   ├─> 16 API key pool management
    │   ├─> Retry logic with backoff
    │   └─> Manifest creation
    │
    ▼
L1 (Standardization)
    ├─> DAG: usdcop_m5__02_l1_standardize.py
    ├─> Input: L0 MinIO data
    ├─> Output: PostgreSQL + MinIO (01-l1-ds-usdcop-standardize)
    ├─> Features:
    │   ├─> Data validation (OHLC integrity)
    │   ├─> Duplicate removal
    │   ├─> Timezone normalization
    │   └─> Schema enforcement
    │
    ▼
L2 (Preparation)
    ├─> DAG: usdcop_m5__03_l2_prepare.py
    ├─> Input: L1 standardized data
    ├─> Output: PostgreSQL + MinIO (02-l2-ds-usdcop-prepare)
    ├─> Features:
    │   ├─> Technical indicators (EMA, SMA, RSI)
    │   ├─> Bollinger Bands
    │   ├─> MACD
    │   └─> Volume analysis
    │
    ▼
L3 (Feature Engineering)
    ├─> DAG: usdcop_m5__04_l3_feature.py
    ├─> Input: L2 prepared data
    ├─> Output: MinIO (03-l3-ds-usdcop-feature)
    ├─> Features:
    │   ├─> Lagged features (t-1, t-2, t-5)
    │   ├─> Rolling statistics
    │   ├─> Correlation analysis
    │   └─> Feature scaling
    │
    ▼
L4 (RL-Ready)
    ├─> DAG: usdcop_m5__05_l4_rlready.py
    ├─> Input: L3 features
    ├─> Output: MinIO (04-l4-ds-usdcop-rlready)
    ├─> Features:
    │   ├─> Episode-based dataset
    │   ├─> Reward calculation
    │   ├─> State/action/reward tuples
    │   └─> Training/validation split
    │
    ▼
L5 (Serving)
    ├─> DAG: usdcop_m5__06_l5_serving.py
    ├─> Input: L4 RL dataset
    ├─> Output: MinIO (05-l5-ds-usdcop-serving)
    ├─> Features:
    │   ├─> Model deployment
    │   ├─> Inference endpoints
    │   ├─> Prediction caching
    │   └─> A/B testing support
    │
    ▼
L6 (Backtesting)
    ├─> DAG: usdcop_m5__07_l6_backtest_referencia.py
    ├─> Input: L5 serving data
    ├─> Output: MinIO (usdcop-l6-backtest)
    ├─> Features:
    │   ├─> Historical simulation
    │   ├─> Performance metrics
    │   ├─> Risk analysis
    │   └─> Trade replay
```

### 3. API Services

All services are FastAPI applications with:
- OpenAPI/Swagger documentation
- CORS support for dashboard
- Health check endpoints
- Structured logging
- Connection pooling

**Trading API** (Port 8000)
```python
Endpoints:
├─> GET /api/health
├─> GET /api/latest/{symbol}
├─> GET /api/candlesticks/{symbol}
├─> GET /api/stats/{symbol}
├─> GET /api/market/health
├─> GET /api/market/historical
└─> GET /api/trading/positions
```

**Analytics API** (Port 8001)
```python
Endpoints:
├─> GET /api/analytics/rl-metrics
├─> GET /api/analytics/performance-kpis
├─> GET /api/analytics/production-gates
├─> GET /api/analytics/session-pnl
├─> GET /api/analytics/risk-metrics
└─> GET /api/analytics/trade-distribution
```

**Pipeline API** (Port 8002)
```python
Endpoints:
├─> GET /api/pipeline/status
├─> GET /api/pipeline/layer/{layer_name}
├─> GET /api/pipeline/health
├─> GET /api/pipeline/data-quality
└─> GET /api/pipeline/manifests
```

**Compliance API** (Port 8003)
```python
Endpoints:
├─> GET /api/compliance/audit-trail
├─> GET /api/compliance/regulations
└─> POST /api/compliance/report
```

### 4. Dashboard (Next.js)

**Port**: 5000
**Technology**: Next.js 15.5.2, React 19, TypeScript

```
app/
├── page.tsx                    # Main dashboard (pipeline status)
├── trading/                    # Trading terminal
│   └── page.tsx
├── ml-analytics/               # RL analytics
│   └── page.tsx
├── diagnostico/                # System diagnostics
│   └── page.tsx
├── api/                        # Next.js API routes
│   ├── pipeline/
│   │   └── consolidated/
│   │       └── route.ts        # Aggregates pipeline data
│   └── health/
│       └── route.ts            # Health check
└── layout.tsx                  # Root layout

components/
├── charts/
│   ├── TradingViewChart.tsx    # Advanced charting
│   ├── OrderFlowChart.tsx      # Volume profile
│   └── RiskMetricsChart.tsx    # Risk visualization
├── pipeline/
│   ├── LayerCard.tsx           # L0-L6 status cards
│   └── HealthIndicator.tsx     # Health status
└── trading/
    ├── OrderForm.tsx           # Order placement
    └── PositionTable.tsx       # Active positions
```

**State Management**:
- **Zustand**: Global state (user preferences, theme)
- **Jotai**: Derived state (computed values)
- **TanStack Query**: Server state (API caching)
- **SWR**: Real-time data (WebSocket sync)

---

## Data Flow

### Real-Time Data Flow (Live Trading)

```
┌─────────────┐
│ Market Opens│ 08:00 AM COT
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│ Step 1: L0 Pipeline Execution (Airflow)                  │
│ ────────────────────────────────────────────────────     │
│  TwelveData API → Parse OHLCV → Validate → PostgreSQL   │
│                                            └─> MinIO     │
│  Duration: ~30-60 seconds                                │
│  Records: 1 bar (5-minute candle)                        │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│ Step 2: RT Orchestrator Detects L0 Completion           │
│ ────────────────────────────────────────────────────     │
│  Polls: SELECT * FROM pipeline_status                    │
│         WHERE pipeline_name LIKE '%L0%'                  │
│         AND status = 'completed'                         │
│  Result: L0 completed at 08:00:45                        │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│ Step 3: RT Collection Starts                            │
│ ────────────────────────────────────────────────────     │
│  Every 5 seconds:                                        │
│    1. Fetch TwelveData API                               │
│    2. Parse JSON response                                │
│    3. Write to PostgreSQL (upsert)                       │
│    4. Publish to Redis channel                           │
│  Duration: 08:01:00 - 12:55:00                           │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│ Step 4: WebSocket Distribution                          │
│ ────────────────────────────────────────────────────     │
│  Redis Pub/Sub:                                          │
│    PUBLISH market_data '{"symbol": "USDCOP", ...}'      │
│  WebSocket Server (Port 8082):                          │
│    ├─> Client 1 (Dashboard)                             │
│    ├─> Client 2 (Mobile app)                            │
│    └─> Client N (External consumer)                     │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│ Step 5: Dashboard Update                                │
│ ────────────────────────────────────────────────────     │
│  React Component:                                        │
│    useWebSocket('ws://localhost:8082')                  │
│  On message:                                             │
│    1. Parse JSON                                         │
│    2. Update Zustand store                               │
│    3. Re-render chart                                    │
│  Latency: < 50ms from API to UI                          │
└──────────────────────────────────────────────────────────┘
```

### Batch Data Flow (Pipeline Processing)

```
┌─────────────────────────────────────────────────────────┐
│ Airflow Scheduler (Cron: */5 * * * * during hours)     │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│ L0: Intelligent Gap Detection                           │
│ ─────────────────────────────────────────────────────   │
│  1. Query PostgreSQL for latest timestamp               │
│  2. Compare with current time                           │
│  3. Detect missing bars                                 │
│  4. Fetch missing data (parallel API calls)             │
│  5. Write to PostgreSQL + MinIO                         │
│  API Keys: Round-robin across 16 keys                   │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│ L1: Standardization & Validation                        │
│ ─────────────────────────────────────────────────────   │
│  1. Read from MinIO (00-raw-usdcop-marketdata)          │
│  2. Validate: open <= high, low <= close                │
│  3. Remove duplicates by timestamp                      │
│  4. Normalize timezone to UTC                           │
│  5. Write to PostgreSQL + MinIO (01-l1-...)             │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│ L2: Technical Indicators                                │
│ ─────────────────────────────────────────────────────   │
│  1. Read from L1 MinIO bucket                           │
│  2. Calculate indicators:                               │
│     ├─> EMA(20), EMA(50)                                │
│     ├─> SMA(10), SMA(20)                                │
│     ├─> RSI(14)                                         │
│     ├─> MACD(12, 26, 9)                                 │
│     └─> Bollinger Bands(20, 2)                          │
│  3. Write to PostgreSQL (for serving)                   │
│  4. Write to MinIO (02-l2-...) for archival             │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│ L3: Feature Engineering                                 │
│ ─────────────────────────────────────────────────────   │
│  1. Read from L2 MinIO bucket                           │
│  2. Create lagged features (t-1, t-2, t-5, t-10)        │
│  3. Rolling statistics (mean, std, min, max)            │
│  4. Price momentum (% change over N periods)            │
│  5. Volume ratios (current vs average)                  │
│  6. Write to MinIO (03-l3-...) as Parquet               │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│ L4: RL Dataset Creation                                 │
│ ─────────────────────────────────────────────────────   │
│  1. Read from L3 MinIO bucket                           │
│  2. Define episodes (trading sessions)                  │
│  3. Calculate rewards:                                  │
│     reward = (close_t+1 - close_t) * position           │
│  4. Create state vectors (normalized features)          │
│  5. Train/validation split (80/20)                      │
│  6. Write to MinIO (04-l4-...) as Parquet               │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│ L5: Model Serving                                       │
│ ─────────────────────────────────────────────────────   │
│  1. Read from L4 MinIO bucket                           │
│  2. Load trained RL model (from MLflow)                 │
│  3. Generate predictions for latest data                │
│  4. Cache predictions in Redis                          │
│  5. Expose via API endpoint                             │
│  6. Write predictions to MinIO (05-l5-...)              │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│ L6: Backtesting & Validation                            │
│ ─────────────────────────────────────────────────────   │
│  1. Read from L5 MinIO bucket                           │
│  2. Simulate trades based on predictions                │
│  3. Calculate performance metrics:                      │
│     ├─> Sharpe Ratio                                    │
│     ├─> Sortino Ratio                                   │
│     ├─> Max Drawdown                                    │
│     ├─> Win Rate                                        │
│     └─> Total P&L                                       │
│  4. Generate report PDF                                 │
│  5. Write results to MinIO (usdcop-l6-backtest)         │
│  6. Update dashboard metrics in PostgreSQL              │
└─────────────────────────────────────────────────────────┘
```

---

## Service Layer

### Microservices Communication

```
┌─────────────────────────────────────────────────────────┐
│                 Service Communication                    │
└─────────────────────────────────────────────────────────┘

Dashboard (Next.js)
    │
    ├─> HTTP GET → Trading API (8000)
    │   └─> Response: Market data JSON
    │
    ├─> HTTP GET → Analytics API (8001)
    │   └─> Response: RL metrics JSON
    │
    ├─> HTTP GET → Pipeline API (8002)
    │   └─> Response: Layer health JSON
    │
    ├─> HTTP GET → Compliance API (8003)
    │   └─> Response: Audit trail JSON
    │
    └─> WebSocket → WebSocket Service (8082)
        └─> Stream: Real-time price updates

All APIs → PostgreSQL (asyncpg pool)
All APIs → Redis (redis-py)
RT Orchestrator → TwelveData API (HTTPS)
Airflow → MinIO (boto3 S3 client)
```

### API Gateway (Future Enhancement)

Currently, services are accessed directly. For production, consider:
- **NGINX** as reverse proxy (already in docker-compose)
- **Kong** or **Traefik** for advanced routing
- **API rate limiting** per client
- **JWT authentication** for secure access

---

## Storage Layer

### Database Schema (PostgreSQL + TimescaleDB)

**Hypertable: `usdcop_m5_ohlcv`**
```sql
CREATE TABLE usdcop_m5_ohlcv (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    open NUMERIC(20,8),
    high NUMERIC(20,8),
    low NUMERIC(20,8),
    close NUMERIC(20,8),
    volume NUMERIC(20,2),
    source TEXT,  -- 'L0', 'realtime', 'backfill'
    ema_20 NUMERIC(20,8),
    ema_50 NUMERIC(20,8),
    sma_10 NUMERIC(20,8),
    sma_20 NUMERIC(20,8),
    rsi NUMERIC(10,2),
    macd NUMERIC(20,8),
    macd_signal NUMERIC(20,8),
    bb_upper NUMERIC(20,8),
    bb_middle NUMERIC(20,8),
    bb_lower NUMERIC(20,8),
    PRIMARY KEY (time, symbol)
);

-- Convert to hypertable (TimescaleDB)
SELECT create_hypertable('usdcop_m5_ohlcv', 'time');

-- Automatic chunking by time (7 days per chunk)
SELECT set_chunk_time_interval('usdcop_m5_ohlcv', INTERVAL '7 days');

-- Compression after 30 days
ALTER TABLE usdcop_m5_ohlcv SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'time DESC',
    timescaledb.compress_segmentby = 'symbol'
);

SELECT add_compression_policy('usdcop_m5_ohlcv', INTERVAL '30 days');

-- Data retention (drop chunks older than 1 year)
SELECT add_retention_policy('usdcop_m5_ohlcv', INTERVAL '1 year');
```

**Table: `pipeline_status`**
```sql
CREATE TABLE pipeline_status (
    id SERIAL PRIMARY KEY,
    pipeline_name TEXT NOT NULL,
    pipeline_type TEXT NOT NULL,  -- 'L0', 'L1', ... 'L6'
    status TEXT NOT NULL,  -- 'running', 'completed', 'failed'
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    records_processed INTEGER,
    error_message TEXT
);

CREATE INDEX idx_pipeline_status_name ON pipeline_status(pipeline_name);
CREATE INDEX idx_pipeline_status_type ON pipeline_status(pipeline_type);
CREATE INDEX idx_pipeline_status_started ON pipeline_status(started_at DESC);
```

**Table: `rl_metrics`**
```sql
CREATE TABLE rl_metrics (
    id SERIAL PRIMARY KEY,
    metric_date DATE NOT NULL,
    symbol TEXT NOT NULL,
    win_rate NUMERIC(5,2),
    sharpe_ratio NUMERIC(10,4),
    sortino_ratio NUMERIC(10,4),
    max_drawdown NUMERIC(10,4),
    total_trades INTEGER,
    total_pnl NUMERIC(20,2),
    avg_trade_duration_minutes INTEGER
);
```

### MinIO Bucket Structure

```
00-raw-usdcop-marketdata/
    └── USDCOP/
        └── M5/
            └── 2025/
                └── 10/
                    └── 22/
                        ├── ohlcv_20251022_080000_080500.parquet
                        ├── ohlcv_20251022_080500_081000.parquet
                        └── manifest.json

01-l1-ds-usdcop-standardize/
    └── USDCOP/
        └── M5/
            └── 2025/
                └── 10/
                    └── 22/
                        ├── standardized_20251022.parquet
                        └── manifest.json

02-l2-ds-usdcop-prepare/
    └── ... (similar structure)

03-l3-ds-usdcop-feature/
    └── ... (similar structure)

04-l4-ds-usdcop-rlready/
    └── episodes/
        └── 2025/
            └── 10/
                ├── episode_20251022_session1.parquet
                └── manifest.json

05-l5-ds-usdcop-serving/
    └── predictions/
        └── 2025/
            └── 10/
                ├── predictions_20251022.parquet
                └── manifest.json

usdcop-l6-backtest/
    └── results/
        └── 2025/
            └── 10/
                ├── backtest_report_20251022.json
                ├── backtest_report_20251022.pdf
                └── manifest.json
```

**Manifest File Format**:
```json
{
    "bucket": "00-raw-usdcop-marketdata",
    "prefix": "USDCOP/M5/2025/10/22",
    "files": [
        {
            "filename": "ohlcv_20251022_080000_080500.parquet",
            "size_bytes": 1024,
            "created_at": "2025-10-22T08:05:00Z",
            "records": 1,
            "checksum": "sha256:abc123..."
        }
    ],
    "metadata": {
        "symbol": "USDCOP",
        "timeframe": "5m",
        "layer": "L0",
        "date": "2025-10-22"
    }
}
```

---

## Real-Time Processing

### WebSocket Protocol

**Connection**:
```javascript
// Client-side
const ws = new WebSocket('ws://localhost:8082/ws/market-data');

ws.onopen = () => {
    console.log('Connected to RT feed');
    // Subscribe to symbol
    ws.send(JSON.stringify({
        action: 'subscribe',
        symbols: ['USDCOP']
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Price update:', data);
};
```

**Message Format**:
```json
{
    "type": "market_data",
    "timestamp": "2025-10-22T08:05:30Z",
    "symbol": "USDCOP",
    "data": {
        "price": 4350.50,
        "open": 4348.00,
        "high": 4352.00,
        "low": 4347.50,
        "close": 4350.50,
        "volume": 1234567,
        "bid": 4350.25,
        "ask": 4350.75,
        "spread": 0.50
    }
}
```

### Redis Pub/Sub

**Publisher (RT Orchestrator)**:
```python
import redis
import json

r = redis.Redis(host='redis', port=6379, password='redis123')

def publish_market_data(symbol: str, data: dict):
    message = {
        'type': 'market_data',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'symbol': symbol,
        'data': data
    }
    r.publish('market_data_channel', json.dumps(message))
```

**Subscriber (WebSocket Service)**:
```python
import redis
import asyncio

r = redis.Redis(host='redis', port=6379, password='redis123')
pubsub = r.pubsub()
pubsub.subscribe('market_data_channel')

async def listen_and_broadcast():
    for message in pubsub.listen():
        if message['type'] == 'message':
            data = json.loads(message['data'])
            # Broadcast to all WebSocket clients
            await broadcast_to_clients(data)
```

---

## Design Decisions

### ADR-001: Why Separate RT Orchestrator from L0 Pipeline?

**Context**: Initial design had L0 pipeline handle both batch and real-time data.

**Problem**:
- L0 pipeline runs every 5 minutes (not truly real-time)
- Airflow has overhead (~5-10 seconds to start task)
- Mixing batch and streaming causes complexity

**Decision**: Create dedicated RT Orchestrator service.

**Consequences**:
- ✅ Pros: True real-time (<5s latency), simpler L0 logic, easier to debug
- ❌ Cons: More services to manage, potential data duplication
- Mitigation: RT Orchestrator waits for L0 completion, uses upsert strategy

### ADR-002: Why PostgreSQL + MinIO Dual Storage?

**Context**: Need to serve dashboard queries fast AND store all history.

**Problem**:
- PostgreSQL great for queries, but expensive for massive storage
- MinIO cheap for storage, but slower for random queries
- S3-only would require complex querying (Athena, Presto)

**Decision**: Use both - PostgreSQL for hot data, MinIO for cold.

**Consequences**:
- ✅ Pros: Fast dashboard, cheap archival, regulatory compliance
- ❌ Cons: Data duplication, need sync mechanism
- Mitigation: Airflow writes to both atomically, periodic validation

### ADR-003: Why TimescaleDB over InfluxDB?

**Context**: Need time-series database for OHLCV data.

**Alternatives**:
1. **InfluxDB**: Purpose-built for time-series, fast writes
2. **TimescaleDB**: PostgreSQL extension, SQL queries
3. **ClickHouse**: Columnar, OLAP-optimized

**Decision**: TimescaleDB

**Rationale**:
- SQL familiarity (easier for team)
- PostgreSQL ecosystem (pgAdmin, Grafana connectors)
- Automatic chunking and compression
- Hypertables = transparent time partitioning

**Consequences**:
- ✅ Pros: Standard SQL, great tooling, good performance
- ❌ Cons: Not as fast as specialized TSDB for pure time-series
- Trade-off: Acceptable for our scale (<100k bars/day)

### ADR-004: Why 16 API Keys?

**Context**: TwelveData free tier = 8 credits/day, need ~288 calls/day (72 bars * 4 retries).

**Problem**: Single API key insufficient for reliable data collection.

**Decision**: Support 16 API keys (2 groups of 8) with round-robin.

**Consequences**:
- ✅ Pros: 128 credits/day (44% buffer), fault tolerance, faster gap filling
- ❌ Cons: Complex key management, potential API abuse concerns
- Mitigation: Respect rate limits, implement exponential backoff

### ADR-005: Why Airflow LocalExecutor over CeleryExecutor?

**Context**: Need task orchestration, choice between executors.

**Alternatives**:
1. **LocalExecutor**: Tasks run in scheduler process (multi-threading)
2. **CeleryExecutor**: Tasks run in separate worker processes (distributed)

**Decision**: LocalExecutor for now, easy upgrade to Celery later.

**Rationale**:
- Simpler deployment (no Celery workers)
- Our DAGs are lightweight (< 5 min each)
- Single-node deployment sufficient for current scale

**Consequences**:
- ✅ Pros: Simpler architecture, fewer moving parts
- ❌ Cons: Limited parallelism (bound by single node)
- Future: Switch to CeleryExecutor when scale requires

---

## Comparison: Old vs New Architecture

### Old Architecture (Pre-October 2025)

```
┌─────────────┐
│  Dashboard  │
└──────┬──────┘
       │
       ▼
┌─────────────┐      ┌─────────────┐
│ Trading API │─────▶│ PostgreSQL  │
│ (Port 8000) │      │ (No TS)     │
└─────────────┘      └─────────────┘
       │
       ▼
┌─────────────┐
│  Hardcoded  │
│    Data     │
│  (No real   │
│   pipeline) │
└─────────────┘
```

**Limitations**:
- ❌ No real data pipeline
- ❌ Hardcoded values in dashboard
- ❌ No real-time data
- ❌ Manual data updates
- ❌ No archival storage
- ❌ Single API, no separation of concerns

### New Architecture (Current - October 2025)

```
┌────────────────────────────────────────────────────────┐
│                   Dashboard (Next.js)                   │
└────┬───────────────────────────────────────────────┬───┘
     │                                               │
     ▼                                               ▼
┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
│Trading API │  │Analytics   │  │Pipeline    │  │Compliance  │
│(Port 8000) │  │API (8001)  │  │API (8002)  │  │API (8003)  │
└────┬───────┘  └────┬───────┘  └────┬───────┘  └────┬───────┘
     │               │               │               │
     └───────┬───────┴───────┬───────┴───────┬───────┘
             │               │               │
             ▼               ▼               ▼
     ┌──────────────────────────────────────────────┐
     │         PostgreSQL + TimescaleDB             │
     └──────────────────────────────────────────────┘
                            │
     ┌──────────────────────┼──────────────────────┐
     │                      │                      │
     ▼                      ▼                      ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Airflow   │    │      RT       │    │    MinIO    │
│ L0-L6 DAGs  │    │ Orchestrator  │    │ Data Lake   │
│(Port 8080)  │    │ (Port 8085)   │    │ (Port 9000) │
└─────────────┘    └──────────────┘    └─────────────┘
     │                      │                      │
     └──────────────────────┴──────────────────────┘
                            │
                            ▼
                  ┌──────────────────┐
                  │  TwelveData API  │
                  │  (16 API Keys)   │
                  └──────────────────┘
```

**Improvements**:
- ✅ **100% Dynamic Data**: No hardcoded values
- ✅ **7-Layer Pipeline**: L0-L6 medallion architecture
- ✅ **Real-Time Collection**: RT Orchestrator with WebSocket
- ✅ **Dual Storage**: PostgreSQL (hot) + MinIO (cold)
- ✅ **4 Specialized APIs**: Separation of concerns
- ✅ **Intelligent Gap Detection**: Auto-fill missing data
- ✅ **16 API Key Pool**: High availability and speed
- ✅ **Market Hours Aware**: Only runs during trading hours
- ✅ **Dependency Management**: RT waits for L0
- ✅ **Observability**: Prometheus, Grafana, structured logs

**Migration Impact**:
- Dashboard: 0% code changes (APIs backward compatible)
- Backend: 100% rewritten (FastAPI microservices)
- Data: PostgreSQL schema upgraded (new columns)
- Infrastructure: Docker Compose updated (new services)
- Downtime: Zero (blue-green deployment possible)

---

## Future Enhancements

### Short-Term (Next 3 Months)

1. **Kubernetes Deployment**
   - Helm charts for each service
   - Horizontal Pod Autoscaling
   - Persistent Volume Claims for data

2. **API Gateway**
   - Single entry point (Kong or Traefik)
   - JWT authentication
   - Rate limiting per client

3. **Advanced Monitoring**
   - Distributed tracing (OpenTelemetry)
   - Custom Grafana dashboards
   - PagerDuty integration for alerts

### Mid-Term (3-6 Months)

1. **Multi-Symbol Support**
   - Extend beyond USD/COP
   - Support for USDBRL, USDMXN, etc.
   - Symbol-specific configurations

2. **ML Model Management**
   - MLflow integration for model registry
   - A/B testing framework
   - Automated retraining pipeline

3. **Advanced Backtesting**
   - Walk-forward analysis
   - Monte Carlo simulations
   - Strategy optimization

### Long-Term (6+ Months)

1. **Cloud Deployment**
   - AWS/Azure/GCP migration
   - Managed services (RDS, ElastiCache, S3)
   - Auto-scaling infrastructure

2. **Multi-Tenancy**
   - Per-user workspaces
   - Isolated data and models
   - Usage-based billing

3. **Mobile App**
   - React Native dashboard
   - Push notifications for alerts
   - Offline mode with sync

---

## Conclusion

The USDCOP Trading System implements a robust, scalable architecture for real-time algorithmic trading. Key strengths:

1. **Separation of Concerns**: Dedicated services for ingestion, processing, serving
2. **Resilience**: Multiple API keys, intelligent retries, fallback mechanisms
3. **Performance**: TimescaleDB for fast queries, Redis for real-time, MinIO for scale
4. **Observability**: Comprehensive logging, metrics, and health checks
5. **Maintainability**: Clear separation, documented decisions, standard patterns

This architecture supports both current production needs and future scaling to millions of bars per day across multiple symbols.

**For operational procedures, see:** `docs/RUNBOOK.md`
**For development guidelines, see:** `docs/DEVELOPMENT.md`
**For migration instructions, see:** `docs/MIGRATION_GUIDE.md`
