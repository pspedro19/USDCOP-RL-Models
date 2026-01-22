# Feast Feature Store Integration Guide
## USD/COP RL Trading System

**Contract**: FEAST-29, INTDOC-08
**Version**: 1.0.0
**Last Updated**: 2026-01-17

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Configuration](#configuration)
4. [Feature Views](#feature-views)
5. [Materialization](#materialization)
6. [Inference Integration](#inference-integration)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

---

## Overview

Feast (Feature Store) is used in the USDCOP Trading System to:

- **Serve features in real-time** for inference
- **Ensure training/inference parity** with consistent feature values
- **Materialize features** from PostgreSQL to Redis for low-latency access
- **Provide fallback mechanisms** when the feature store is unavailable

### Key Components

| Component | Purpose | Location |
|-----------|---------|----------|
| Feature Store Config | Feast configuration | `feature_repo/feature_store.yaml` |
| Feature Views | Feature definitions | `feature_repo/features.py` |
| Materialize DAG | Sync PostgreSQL → Redis | `airflow/dags/l1b_feast_materialize.py` |
| Feast Service | Inference integration | `src/feature_store/feast_service.py` |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEAST ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐  │
│  │  PostgreSQL   │───▶│    Parquet    │───▶│     Redis     │  │
│  │ (Offline Store)│    │   (Files)     │    │ (Online Store) │  │
│  └───────────────┘    └───────────────┘    └───────────────┘  │
│         │                                          │            │
│         │          materialize-incremental         │            │
│         └──────────────────────────────────────────┘            │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                   FEATURE VIEWS                            │ │
│  │  • technical_features (RSI, ATR, ADX, log returns)        │ │
│  │  • macro_features (DXY, VIX, EMBI, Brent, etc.)           │ │
│  │  • state_features (position, time_normalized)              │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                   FEATURE SERVICE                          │ │
│  │  observation_15d: 15 features in canonical order           │ │
│  │  Contract: CTR-FEATURE-001                                 │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration

### Feature Store Configuration

**File**: `feature_repo/feature_store.yaml`

```yaml
project: usdcop_trading
registry: data/registry.db
provider: local

online_store:
  type: redis
  connection_string: redis://redis:6379

offline_store:
  type: file  # Development
  # For production:
  # type: postgres
  # host: postgres
  # port: 5432
  # database: usdcop_trading
  # user: ${POSTGRES_USER}
  # password: ${POSTGRES_PASSWORD}

entity_key_serialization_version: 2
```

### Environment Variables

```bash
# Required for Feast
FEAST_REDIS_HOST=redis
FEAST_REDIS_PORT=6379
FEAST_OFFLINE_STORE_TYPE=file

# For production PostgreSQL offline store
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=usdcop_trading
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
```

---

## Feature Views

### Entity Definition

```python
# Entity representing the trading symbol
usdcop_symbol = Entity(
    name="symbol",
    value_type=ValueType.STRING,
    description="Trading symbol (e.g., USDCOP)"
)
```

### Technical Features View

```python
technical_features = FeatureView(
    name="technical_features",
    entities=[usdcop_symbol],
    ttl=timedelta(hours=24),
    schema=[
        Field(name="log_ret_5m", dtype=Float32),
        Field(name="log_ret_1h", dtype=Float32),
        Field(name="log_ret_4h", dtype=Float32),
        Field(name="rsi_9", dtype=Float32),
        Field(name="atr_pct", dtype=Float32),
        Field(name="adx_14", dtype=Float32),
    ],
    online=True,
    source=technical_source,
)
```

### Macro Features View

```python
macro_features = FeatureView(
    name="macro_features",
    entities=[usdcop_symbol],
    ttl=timedelta(hours=24),
    schema=[
        Field(name="dxy_z", dtype=Float32),
        Field(name="dxy_change_1d", dtype=Float32),
        Field(name="vix_z", dtype=Float32),
        Field(name="embi_z", dtype=Float32),
        Field(name="brent_change_1d", dtype=Float32),
        Field(name="rate_spread", dtype=Float32),
        Field(name="usdmxn_change_1d", dtype=Float32),
    ],
    online=True,
    source=macro_source,
)
```

### Feature Service

```python
observation_15d = FeatureService(
    name="observation_15d",
    features=[
        technical_features,
        macro_features,
        state_features,
    ],
    description="15-dimensional observation for RL inference"
)
```

---

## Materialization

### Materialization DAG

**File**: `airflow/dags/l1b_feast_materialize.py`

The materialization DAG runs after L1 features are computed:

```
L0 (Raw Data) → L1 (Features) → L1b (Materialize) → L5 (Inference)
```

### Running Materialization

```bash
# Manual materialization
cd feature_repo
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")

# Via Airflow
airflow dags trigger v3.l1b_feast_materialize
```

### Materialization Schedule

| Trigger | Schedule | Description |
|---------|----------|-------------|
| After L1 | Sensor | Waits for L1 feature completion |
| Fallback | `*/15 13-18 * * 1-5` | Every 15 min during trading hours |

---

## Inference Integration

### FeastInferenceService

**File**: `src/feature_store/feast_service.py`

```python
from src.feature_store.feast_service import FeastInferenceService

# Initialize service
feast_service = FeastInferenceService()

# Get features for inference
features = feast_service.get_online_features(
    entity_rows=[{"symbol": "USDCOP"}],
    feature_service="observation_15d"
)

# Returns dict with all 15 features
# {'log_ret_5m': 0.0012, 'rsi_9': 45.2, ...}
```

### Fallback Mechanism

If Feast is unavailable, the system falls back to `CanonicalFeatureBuilder`:

```python
try:
    features = feast_service.get_online_features(...)
except FeastUnavailableError:
    # Fallback to direct calculation
    builder = CanonicalFeatureBuilder.for_inference()
    features = builder.build_observation(timestamp)
```

---

## Troubleshooting

### Common Issues

#### 1. Redis Connection Failed

**Symptom**: `ConnectionError: Error connecting to Redis`

**Solution**:
```bash
# Check Redis is running
docker-compose ps redis

# Verify Redis connectivity
docker exec -it usdcop-redis redis-cli ping

# Check Redis password
docker exec -it usdcop-redis redis-cli -a ${REDIS_PASSWORD} ping
```

#### 2. Features Not Materialized

**Symptom**: `FeatureNotFoundError` or stale feature values

**Solution**:
```bash
# Check materialization status
cd feature_repo
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")

# Verify features in Redis
docker exec -it usdcop-redis redis-cli keys "feast:*"
```

#### 3. Feature Schema Mismatch

**Symptom**: `ValueError: Feature count mismatch`

**Solution**:
```bash
# Re-apply feature definitions
cd feature_repo
feast apply

# Verify feature views
feast feature-views list
```

#### 4. Offline Store Connection Issues

**Symptom**: `OperationalError: connection refused`

**Solution**:
```bash
# Check PostgreSQL connectivity
docker-compose ps postgres
pg_isready -h localhost -p 5432

# Verify connection string in feature_store.yaml
```

### Diagnostic Commands

```bash
# List all entities
feast entities list

# List all feature views
feast feature-views list

# List all feature services
feast feature-services list

# Show registry contents
feast registry-dump

# Validate feature definitions
feast validate
```

---

## Best Practices

### 1. Feature Naming Convention

Follow the SSOT feature contract:
- Use snake_case: `log_ret_5m`, `rsi_9`
- Include period in name: `adx_14`, `atr_pct`
- Use z-score suffix for normalized: `dxy_z`, `vix_z`

### 2. TTL Configuration

- **Technical features**: 24 hours (intraday data)
- **Macro features**: 24 hours (daily data)
- **State features**: 1 hour (position changes frequently)

### 3. Materialization Frequency

- Run after every L1 feature pipeline completion
- Use sensors to wait for upstream DAGs
- Implement idempotent materialization

### 4. Error Handling

Always implement fallback mechanisms:

```python
def get_features_with_fallback(symbol: str) -> dict:
    try:
        return feast_service.get_online_features(symbol)
    except Exception as e:
        logger.warning(f"Feast unavailable: {e}, using fallback")
        return canonical_builder.build_features(symbol)
```

### 5. Monitoring

Monitor these metrics:
- Materialization latency
- Feature freshness (time since last update)
- Redis memory usage
- Feature retrieval latency

---

## Related Documentation

- [Feature Contract (CTR-FEATURE-001)](../src/core/contracts/feature_contract.py)
- [CanonicalFeatureBuilder](../src/feature_store/builders.py)
- [L1b Materialize DAG](../airflow/dags/l1b_feast_materialize.py)
- [Feast Troubleshooting Runbook](./FEAST_TROUBLESHOOTING_RUNBOOK.md)

---

*Document maintained by USDCOP Trading Team*
