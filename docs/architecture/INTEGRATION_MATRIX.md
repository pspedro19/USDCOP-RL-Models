# Service Integration Matrix
## USD/COP RL Trading System

**Contract**: INTDOC-04
**Version**: 1.0.0
**Last Updated**: 2026-01-17

---

## Overview

This document provides a comprehensive matrix of all service integrations in the USDCOP Trading System.

---

## Service Integration Matrix

### Data Flow Integrations

| Source | Target | Protocol | Port | Purpose | Direction |
|--------|--------|----------|------|---------|-----------|
| Scrapers | PostgreSQL | TCP/PGSQL | 5432 | OHLCV ingestion | → |
| PostgreSQL | Feast | TCP/PGSQL | 5432 | Feature source | → |
| Feast | Redis | TCP/Redis | 6379 | Online features | → |
| PostgreSQL | Parquet | File | - | Offline features | → |
| Parquet | DVC | File | - | Dataset versioning | → |
| DVC | MinIO | S3/HTTP | 9000 | Remote storage | ↔ |
| MLflow | MinIO | S3/HTTP | 9000 | Artifacts | ↔ |
| MLflow | PostgreSQL | TCP/PGSQL | 5432 | Backend store | ↔ |

### Service Dependencies

| Service | Depends On | Healthcheck | Critical |
|---------|------------|-------------|----------|
| postgres | - | pg_isready | ✓ |
| redis | - | redis-cli ping | ✓ |
| minio | - | /minio/health/live | ✓ |
| mlflow | minio | /health | ✓ |
| airflow-scheduler | postgres, redis | jobs check | ✓ |
| airflow-webserver | airflow-scheduler | /health | |
| feast-materialize | postgres, redis | - | |
| mlops-inference-api | postgres, redis, mlflow, minio | /health | ✓ |
| backtest-api | postgres, redis, mlflow, minio | /v1/health | ✓ |
| trading-api | postgres, redis | /api/health | |
| analytics-api | postgres | /api/health | |
| dashboard | postgres, redis, trading-api | /api/health | |

---

## Protocol Reference

### PostgreSQL (Port 5432)

| Connection | User | Database | Purpose |
|------------|------|----------|---------|
| Airflow | postgres | usdcop_trading | Metadata, XCom |
| MLflow | postgres | usdcop_trading | Experiment tracking |
| DAGs | postgres | usdcop_trading | OHLCV, Macro data |
| APIs | postgres | usdcop_trading | Trading data |

### Redis (Port 6379)

| Client | Database | Purpose |
|--------|----------|---------|
| Feast | 0 | Online features |
| Airflow Celery | 0 | Task queue |
| Trading API | 0 | Caching, Streams |
| Inference | 0 | Feature retrieval |

### MinIO (Ports 9000, 9001)

| Bucket | Access | Purpose |
|--------|--------|---------|
| dvc-storage | DVC | Dataset versioning |
| mlflow-artifacts | MLflow | Model artifacts |
| 00-raw-* | L0 DAGs | Raw data |
| 01-l1-* | L1 DAGs | Standardized data |
| 99-common-* | All | Shared resources |

---

## API Endpoints

### Internal APIs

| Service | Port | Base Path | Auth |
|---------|------|-----------|------|
| trading-api | 8000 | /api | None |
| analytics-api | 8001 | /api/analytics | None |
| mlops-inference-api | 8090 | / | API Key |
| backtest-api | 8003 | /v1 | None |
| multi-model-api | 8006 | /api | None |

### External APIs (Scrapers)

| Provider | Protocol | Rate Limit | Purpose |
|----------|----------|------------|---------|
| TwelveData | HTTPS | 8 req/min | OHLCV |
| FRED | HTTPS | 500/day | Macro |
| BanRep | HTTPS | - | Colombian data |

---

## Airflow DAG Dependencies

```
L0 (Data Acquisition)
├── l0_ohlcv_realtime
├── l0_macro_unified
└── l0_banrep_scraper

L1 (Feature Engineering)
├── l1_feature_refresh
│   └── depends_on: L0 completion
└── l1b_feast_materialize
    └── depends_on: l1_feature_refresh

L3 (Model Training)
└── l3_model_training
    ├── depends_on: DVC checkout (NEW)
    └── depends_on: Dataset availability

L5 (Inference)
└── l5_multi_model_inference
    ├── depends_on: l1b_feast_materialize (sensor)
    └── depends_on: trading_flags validation
```

---

## Data Contracts

### Feature Contract (CTR-FEATURE-001)

| Index | Feature | Source | Type |
|-------|---------|--------|------|
| 0 | log_ret_5m | L1 | Technical |
| 1 | log_ret_1h | L1 | Technical |
| 2 | log_ret_4h | L1 | Technical |
| 3 | rsi_9 | L1 | Technical |
| 4 | atr_pct | L1 | Technical |
| 5 | adx_14 | L1 | Technical |
| 6 | dxy_z | L0 Macro | Macro |
| 7 | dxy_change_1d | L0 Macro | Macro |
| 8 | vix_z | L0 Macro | Macro |
| 9 | embi_z | L0 Macro | Macro |
| 10 | brent_change_1d | L0 Macro | Macro |
| 11 | rate_spread | L0 Macro | Macro |
| 12 | usdmxn_change_1d | L0 Macro | Macro |
| 13 | position | State | State |
| 14 | time_normalized | State | State |

### Hash Contract (CTR-HASH-001)

| Artifact | Hash Algorithm | Location |
|----------|---------------|----------|
| Dataset | SHA256 | MLflow tags |
| Model | SHA256 | MLflow tags |
| Norm Stats | SHA256 (JSON canonical) | MLflow tags |
| Feature Order | SHA256[:16] | Contract |
| Git Commit | SHA1 (git) | MLflow tags |

---

## Network Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                    Docker Network: usdcop-trading-network       │
│                    Subnet: 172.29.0.0/16                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐           │
│  │Postgres │  │  Redis  │  │  MinIO  │  │ MLflow  │           │
│  │  :5432  │  │  :6379  │  │  :9000  │  │  :5001  │           │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘           │
│       │            │            │            │                  │
│       └────────────┴────────────┴────────────┘                  │
│                         │                                       │
│  ┌──────────────────────┴──────────────────────┐               │
│  │                                              │               │
│  │  ┌─────────────────┐  ┌─────────────────┐   │               │
│  │  │ Airflow         │  │ Inference APIs  │   │               │
│  │  │ :8080           │  │ :8003, :8090    │   │               │
│  │  └─────────────────┘  └─────────────────┘   │               │
│  │                                              │               │
│  │  ┌─────────────────┐  ┌─────────────────┐   │               │
│  │  │ Trading APIs    │  │ Dashboard       │   │               │
│  │  │ :8000, :8001    │  │ :5000           │   │               │
│  │  └─────────────────┘  └─────────────────┘   │               │
│  │                                              │               │
│  └──────────────────────────────────────────────┘               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Sync Mechanisms

| Source | Target | Mechanism | Frequency |
|--------|--------|-----------|-----------|
| PostgreSQL | Feast/Redis | materialize-incremental | After L1 |
| Git | DVC | dvc checkout | Before training |
| DVC | MinIO | dvc push | After pipeline |
| MLflow | MinIO | Automatic | On artifact log |
| Airflow | PostgreSQL | XCom | Per task |

---

## Failure Scenarios

| Scenario | Impact | Fallback |
|----------|--------|----------|
| Redis down | Feast unavailable | CanonicalFeatureBuilder |
| PostgreSQL down | All services fail | No fallback |
| MinIO down | DVC/MLflow artifacts fail | S3 backup |
| MLflow down | Model loading fails | Local model cache |
| Feast stale | Old features used | Warning + fallback |

---

## Monitoring Endpoints

| Service | Health | Metrics |
|---------|--------|---------|
| PostgreSQL | pg_isready | pg_exporter:9187 |
| Redis | redis-cli ping | redis_exporter |
| MinIO | /minio/health/live | /minio/v2/metrics |
| MLflow | /health | - |
| Prometheus | /-/healthy | :9090/metrics |
| Grafana | /api/health | - |

---

*Document maintained by USDCOP Trading Team*
