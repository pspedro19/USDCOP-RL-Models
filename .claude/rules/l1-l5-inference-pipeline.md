# Rule: L1→L5 Inference Pipeline

> Governs the unified feature computation and inference pipeline.
> L1 computes + normalizes features. L5 reads + predicts. No duplication.
> Created: 2026-02-12

---

## Architecture

```
L0 (every 5 min)
    │
    ▼
L1 Feature Refresh (l1_feature_refresh.py)
    │  CanonicalFeatureBuilder.compute_features()
    │  Z-score normalize with production norm_stats
    │  Write FLOAT[18] to inference_ready_nrt
    │  pg_notify('features_ready', ...)
    │
    ▼
L5 Multi-Model Inference (l5_multi_model_inference.py)
    │  FeatureReadySensor(channel='features_ready')
    │  Read FLOAT[18] from inference_ready_nrt
    │  Append state features [position, time_norm]
    │  model.predict(observation, deterministic=True)
    │  Store signal + broadcast
    │
    ▼
Trading Signal
```

### On Model Promotion (manual trigger)
```
L1 Model Promotion (l1_model_promotion.py)
    │  Load approved model's norm_stats + dataset
    │  TRUNCATE inference_ready_nrt
    │  Recompute + normalize all historical features
    │  Batch insert to inference_ready_nrt
    │  Copy norm_stats.json → config/norm_stats.json
    │
    ▼
L1 Feature Refresh picks up new norm_stats on next cycle
```

---

## L1 DAG Inventory (2 DAGs)

| # | DAG ID | File | Purpose | Schedule |
|---|--------|------|---------|----------|
| 1 | `rl_l1_01_feature_refresh` | `l1_feature_refresh.py` | Compute + normalize features → `inference_ready_nrt` | `*/5 13-17 * * 1-5` |
| 2 | `rl_l1_03_model_promotion` | `l1_model_promotion.py` | Populate historical features on model approval | Manual / event |

> Note: `rl_l1_02_feast_materialize` was deleted (2026-02-12). Feast is not used.

---

## Core Table: `inference_ready_nrt`

**Migration**: `database/migrations/038_nrt_tables.sql`

```sql
CREATE TABLE inference_ready_nrt (
    id              BIGSERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL UNIQUE,
    features        FLOAT[] NOT NULL,           -- 18 market features, Z-score normalized
    price           DOUBLE PRECISION NOT NULL,  -- Close price of the bar
    feature_order_hash VARCHAR(64),             -- SHA256 of FEATURE_ORDER list
    norm_stats_hash VARCHAR(64),               -- SHA256 of norm_stats.json
    source          VARCHAR(20) DEFAULT 'nrt',  -- 'nrt' (L1 realtime) or 'historical' (promotion)
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
```

**Trigger**: `trg_notify_features_ready` fires `pg_notify('features_ready', ...)` on every INSERT.

### Who writes:
- **L1 Feature Refresh** — writes `source='nrt'` rows (realtime, every 5 min)
- **L1 Model Promotion** — writes `source='historical'` rows (bulk, on approval)

### Who reads:
- **L5 Multi-Model Inference** — reads latest row for prediction

---

## Native Feature Store (replaces Feast)

The system uses a native feature store with 4 guarantees:

| Guarantee | Component | Location |
|-----------|-----------|----------|
| **Same features** | `CanonicalFeatureBuilder` | `src/feature_store/builders/` |
| **Same order** | `FEATURE_ORDER` + `FEATURE_ORDER_HASH` | `src/core/contracts/feature_contract.py` |
| **Same normalization** | `norm_stats.json` + hash validation | `config/norm_stats.json` |
| **Lineage** | `ProductionContract` (2-vote approval) | `src/core/contracts/production_contract.py` |

### Feature Computation Contract
- `CanonicalFeatureBuilder` uses **Wilder's EMA** for RSI/ATR/ADX (NOT pandas `ewm`)
- Features are computed in canonical order defined by `FEATURE_ORDER`
- `FEATURE_ORDER_HASH` = SHA256 of the ordered feature name list
- Hash is stored alongside features in `inference_ready_nrt.feature_order_hash`
- **If hash mismatches between training and inference → model produces garbage**

### Normalization Contract
- Z-score: `(value - mean) / std`, clipped to [-5, 5]
- `norm_stats.json` contains `{feature_name: {mean, std}}` for each market feature
- Hash of norm_stats is stored in `inference_ready_nrt.norm_stats_hash`
- On model promotion, norm_stats is copied to `config/norm_stats.json` for L1

### Model Promotion Flow
```
L4 passes gates → promotion_proposals table → Dashboard 2-vote approval
    → pg_notify('model_approved') → L1 Model Promotion DAG triggers
    → Historical features recomputed with new norm_stats
    → L1 Feature Refresh loads new norm_stats on next cycle
```

---

## Sensor: `FeatureReadySensor`

**File**: `airflow/dags/sensors/postgres_notify_sensor.py`

Dual-mode sensor for L5 to detect new features:
1. **Primary**: PostgreSQL `LISTEN features_ready` (sub-second latency)
2. **Fallback**: Polling `inference_ready_nrt` WHERE timestamp > last_check (every 30s)

Defaults:
```python
channel = 'features_ready'
fallback_table = 'inference_ready_nrt'
fallback_date_column = 'timestamp'
```

Circuit breaker: switches to polling after 3 consecutive LISTEN failures.

---

## Look-Ahead Bias Prevention

L5 uses **signal from bar N-1** (after close) with **execution at bar N OPEN price**:

```python
# Signal price = close of the feature bar (for signal generation)
# Execution price = OPEN of the NEXT bar (for order placement)
cur.execute("""
    SELECT open FROM usdcop_m5_ohlcv
    WHERE symbol='USD/COP' AND time > %s
    ORDER BY time ASC LIMIT 1
""", (feature_ts,))
```

---

## Deprecated Services

These standalone async services have been absorbed into Airflow DAGs:

| Service | Absorbed Into | Deprecation Date |
|---------|--------------|------------------|
| `src/services/l1_nrt_data_service.py` | `l1_feature_refresh.py` + `l1_model_promotion.py` | 2026-02-12 |
| `src/services/l5_nrt_inference_service.py` | `l5_multi_model_inference.py` (v6.0.0) | 2026-02-12 |

---

## DO NOT

- Do NOT compute features in L5 — L1 is the ONLY feature computation layer
- Do NOT use pandas `ewm()` for RSI/ATR/ADX — use `CanonicalFeatureBuilder` (Wilder's EMA)
- Do NOT normalize features in L5 — L1 writes pre-normalized FLOAT[] to `inference_ready_nrt`
- Do NOT skip `feature_order_hash` validation — hash mismatch = garbage predictions
- Do NOT write to `inference_ready_nrt` from anywhere except L1 DAGs
- Do NOT use the deprecated NRT services — they are kept for reference only
- Do NOT bypass `ProductionContract` for model promotion — requires 2-vote approval
- Do NOT add state features (position, time_norm) to `inference_ready_nrt` — L5 adds them at predict time
