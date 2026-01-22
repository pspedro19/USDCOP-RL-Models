# Database Entity-Relationship Diagram
## USD/COP RL Trading System

**Contract**: PG-29
**Version**: 1.0.0
**Last Updated**: 2026-01-17

---

## Overview

This document describes the database schema and relationships for the USDCOP Trading System PostgreSQL database with TimescaleDB extension.

---

## ER Diagram (Text Representation)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATABASE: usdcop_trading                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         SCHEMA: public                               │   │
│  │                                                                       │   │
│  │  ┌───────────────────────┐        ┌───────────────────────┐         │   │
│  │  │  usdcop_m5_ohlcv     │        │ macro_indicators_daily │         │   │
│  │  │  (TimescaleDB)       │        │                        │         │   │
│  │  ├───────────────────────┤        ├───────────────────────┤         │   │
│  │  │ PK time TIMESTAMPTZ  │        │ PK fecha DATE          │         │   │
│  │  │ PK symbol VARCHAR    │        │    indicator_1 NUMERIC │         │   │
│  │  │    open NUMERIC      │        │    indicator_2 NUMERIC │         │   │
│  │  │    high NUMERIC      │        │    ...                 │         │   │
│  │  │    low NUMERIC       │        │    created_at TIMESTAMP│         │   │
│  │  │    close NUMERIC     │        │    updated_at TIMESTAMP│         │   │
│  │  │    volume NUMERIC    │        └───────────────────────┘         │   │
│  │  │    created_at        │                                           │   │
│  │  └───────────────────────┘                                           │   │
│  │                                                                       │   │
│  │  ┌───────────────────────┐        ┌───────────────────────┐         │   │
│  │  │   model_registry     │───────▶│  model_validations    │         │   │
│  │  ├───────────────────────┤        ├───────────────────────┤         │   │
│  │  │ PK id SERIAL         │        │ PK id SERIAL          │         │   │
│  │  │ UK model_id VARCHAR  │        │ FK model_id VARCHAR   │         │   │
│  │  │    model_version     │        │    validation_type    │         │   │
│  │  │    model_path        │        │    passed BOOLEAN     │         │   │
│  │  │    model_hash        │        │    details JSONB      │         │   │
│  │  │    norm_stats_hash   │        │    created_at         │         │   │
│  │  │    config_hash       │        └───────────────────────┘         │   │
│  │  │    observation_dim   │                                           │   │
│  │  │    feature_order JSONB│                                          │   │
│  │  │    status VARCHAR    │                                           │   │
│  │  │    created_at        │                                           │   │
│  │  └───────────────────────┘                                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         SCHEMA: trading                              │   │
│  │                                                                       │   │
│  │  ┌───────────────────────┐        ┌───────────────────────┐         │   │
│  │  │  model_inferences    │        │    paper_trades       │         │   │
│  │  ├───────────────────────┤        ├───────────────────────┤         │   │
│  │  │ PK id SERIAL         │        │ PK id SERIAL          │         │   │
│  │  │    timestamp         │◀──────▶│ FK inference_id       │         │   │
│  │  │    model_id VARCHAR  │        │    model_id VARCHAR   │         │   │
│  │  │    bar_number INT    │        │    signal VARCHAR     │         │   │
│  │  │    raw_action DECIMAL│        │    entry_price DECIMAL│         │   │
│  │  │    signal VARCHAR    │        │    exit_price DECIMAL │         │   │
│  │  │    confidence DECIMAL│        │    pnl_pct DECIMAL    │         │   │
│  │  │    latency_ms DECIMAL│        │    status VARCHAR     │         │   │
│  │  │    observation_hash  │        │    created_at         │         │   │
│  │  │    state_features    │        └───────────────────────┘         │   │
│  │  │    current_price     │                                           │   │
│  │  │    created_at        │                                           │   │
│  │  └───────────────────────┘                                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         SCHEMA: dw                                   │   │
│  │                                                                       │   │
│  │  ┌───────────────────────┐        ┌───────────────────────┐         │   │
│  │  │    dim_strategy      │───────▶│ fact_strategy_signals │         │   │
│  │  ├───────────────────────┤        ├───────────────────────┤         │   │
│  │  │ PK strategy_id SERIAL│        │ PK signal_id SERIAL   │         │   │
│  │  │ UK strategy_code     │        │ FK strategy_id INT    │         │   │
│  │  │    strategy_name     │        │    timestamp_utc      │         │   │
│  │  │    strategy_type     │        │    signal VARCHAR     │         │   │
│  │  │    model_id VARCHAR  │        │    confidence DECIMAL │         │   │
│  │  │    is_active BOOLEAN │        │    entry_price DECIMAL│         │   │
│  │  │    created_at        │        │    reasoning TEXT     │         │   │
│  │  └───────────────────────┘        │    created_at         │         │   │
│  │                                   └───────────────────────┘         │   │
│  │                                                                       │   │
│  │  ┌───────────────────────┐                                           │   │
│  │  │  feature_snapshots   │                                           │   │
│  │  │  (TimescaleDB)       │                                           │   │
│  │  ├───────────────────────┤                                           │   │
│  │  │ PK timestamp         │                                           │   │
│  │  │    symbol VARCHAR    │                                           │   │
│  │  │    features JSONB    │                                           │   │
│  │  │    norm_stats_hash   │                                           │   │
│  │  │    created_at        │                                           │   │
│  │  └───────────────────────┘                                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         SCHEMA: config                               │   │
│  │                                                                       │   │
│  │  ┌───────────────────────┐        ┌───────────────────────┐         │   │
│  │  │      models          │        │    feature_flags      │         │   │
│  │  ├───────────────────────┤        ├───────────────────────┤         │   │
│  │  │ PK model_id VARCHAR  │        │ PK flag_name VARCHAR  │         │   │
│  │  │    model_name        │        │    flag_value BOOLEAN │         │   │
│  │  │    model_type        │        │    description TEXT   │         │   │
│  │  │    model_path        │        │    updated_at         │         │   │
│  │  │    version VARCHAR   │        └───────────────────────┘         │   │
│  │  │    enabled BOOLEAN   │                                           │   │
│  │  │    is_production     │                                           │   │
│  │  │    threshold_long    │                                           │   │
│  │  │    threshold_short   │                                           │   │
│  │  │    created_at        │                                           │   │
│  │  │    updated_at        │                                           │   │
│  │  └───────────────────────┘                                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         SCHEMA: staging                              │   │
│  │                                                                       │   │
│  │  ┌───────────────────────┐        ┌───────────────────────┐         │   │
│  │  │    ohlcv_staging     │        │   macro_staging       │         │   │
│  │  ├───────────────────────┤        ├───────────────────────┤         │   │
│  │  │    (temporary data)  │        │    (temporary data)   │         │   │
│  │  └───────────────────────┘        └───────────────────────┘         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         SCHEMA: audit                                │   │
│  │                                                                       │   │
│  │  ┌───────────────────────┐                                           │   │
│  │  │    change_log        │                                           │   │
│  │  ├───────────────────────┤                                           │   │
│  │  │ PK id SERIAL         │                                           │   │
│  │  │    table_name        │                                           │   │
│  │  │    operation VARCHAR │                                           │   │
│  │  │    old_data JSONB    │                                           │   │
│  │  │    new_data JSONB    │                                           │   │
│  │  │    changed_by        │                                           │   │
│  │  │    changed_at        │                                           │   │
│  │  └───────────────────────┘                                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Table Details

### usdcop_m5_ohlcv (TimescaleDB Hypertable)

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| time | TIMESTAMPTZ | PK, NOT NULL | Bar timestamp (UTC) |
| symbol | VARCHAR(20) | PK, NOT NULL | Trading symbol |
| open | NUMERIC(12,4) | | Opening price |
| high | NUMERIC(12,4) | | High price |
| low | NUMERIC(12,4) | | Low price |
| close | NUMERIC(12,4) | | Closing price |
| volume | NUMERIC(20,2) | | Trading volume |
| created_at | TIMESTAMPTZ | DEFAULT NOW() | Record creation time |

**Indexes**:
- `idx_ohlcv_symbol_time` on (symbol, time DESC)

**Hypertable Settings**:
- chunk_time_interval: 7 days
- compression: enabled after 30 days
- retention: 365 days

### macro_indicators_daily

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| fecha | DATE | PK | Indicator date |
| fxrt_index_dxy_usa_d_dxy | NUMERIC | | DXY Index |
| volt_vix_usa_d_vix | NUMERIC | | VIX Index |
| crsk_spread_embi_col_d_embi | NUMERIC | | EMBI Colombia |
| comm_oil_brent_glb_d_brent | NUMERIC | | Brent Oil |
| finc_bond_yield10y_usa_d_ust10y | NUMERIC | | US 10Y Yield |
| fxrt_spot_usdmxn_mex_d_usdmxn | NUMERIC | | USD/MXN |
| ... | | | 37 indicators total |
| created_at | TIMESTAMPTZ | DEFAULT NOW() | |
| updated_at | TIMESTAMPTZ | DEFAULT NOW() | |

### model_registry

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | SERIAL | PK | Auto-increment ID |
| model_id | VARCHAR(100) | UNIQUE, NOT NULL | Model identifier |
| model_version | VARCHAR(50) | NOT NULL | Version string |
| model_path | VARCHAR(500) | NOT NULL | Path to model file |
| model_hash | VARCHAR(64) | | SHA256 hash |
| norm_stats_hash | VARCHAR(64) | | Normalization stats hash |
| config_hash | VARCHAR(64) | | Configuration hash |
| observation_dim | INT | DEFAULT 15 | Observation dimension |
| action_space | INT | DEFAULT 3 | Action space size |
| feature_order | JSONB | | Feature names in order |
| validation_metrics | JSONB | | Training metrics |
| status | VARCHAR(20) | DEFAULT 'registered' | Model status |
| created_at | TIMESTAMPTZ | DEFAULT NOW() | |

---

## Relationships Summary

| Parent Table | Child Table | Relationship | On Delete |
|--------------|-------------|--------------|-----------|
| model_registry | model_validations | 1:N | CASCADE |
| dim_strategy | fact_strategy_signals | 1:N | SET NULL |
| model_inferences | paper_trades | 1:1 | CASCADE |
| config.models | trading.model_inferences | 1:N | - |

---

## Schema Ownership

| Schema | Purpose | Owner |
|--------|---------|-------|
| public | Core trading data | postgres |
| trading | Inference and trades | trading_app |
| dw | Data warehouse | analytics |
| config | Configuration tables | admin |
| staging | Temporary data | etl |
| audit | Audit logs | auditor |

---

*Document maintained by USDCOP Trading Team*
