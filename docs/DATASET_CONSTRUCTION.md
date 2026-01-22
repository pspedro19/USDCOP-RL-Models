# Dataset Construction Guide

**Document ID**: P0-04
**Version**: 1.0.0
**Last Updated**: 2026-01-17
**Status**: Active

---

## Table of Contents

1. [Overview](#1-overview)
2. [Data Flow Architecture](#2-data-flow-architecture)
3. [Feature Order (SSOT)](#3-feature-order-ssot)
4. [Data Sources](#4-data-sources)
5. [Temporal Joins](#5-temporal-joins)
6. [Trading Hours Filter](#6-trading-hours-filter)
7. [Dataset Versioning](#7-dataset-versioning)
8. [Normalization Statistics](#8-normalization-statistics)
9. [Reproducibility Checklist](#9-reproducibility-checklist)

---

## 1. Overview

This document describes the complete process for constructing training datasets for the USD/COP reinforcement learning trading system. It serves as the authoritative reference for:

- **Data Engineers**: Understanding the data pipeline architecture
- **ML Engineers**: Ensuring training/inference feature parity
- **DevOps**: Maintaining data infrastructure
- **Auditors**: Verifying data lineage and reproducibility

The dataset construction process transforms raw market data and macroeconomic indicators into a 15-dimensional observation space used by PPO agents to make trading decisions.

### Key Principles

1. **Single Source of Truth (SSOT)**: All feature definitions originate from `src/core/contracts/feature_contract.py`
2. **Temporal Integrity**: Strict point-in-time correctness using backward-looking joins
3. **Reproducibility**: All transformations are versioned and logged to MLflow
4. **Data Quality**: Bounded forward-fill with per-indicator limits

---

## 2. Data Flow Architecture

The dataset construction pipeline follows a layered DAG architecture:

```
+------------------------------------------------------------------+
|                        DATA FLOW DIAGRAM                          |
+------------------------------------------------------------------+

                     LAYER 0 (L0) - Raw Data Ingestion
+------------------------------------------------------------------+
|                                                                    |
|   +-------------+    +-------------+    +-----------+             |
|   |   FRED API  |    | TwelveData  |    |  BanRep   |             |
|   | (14 vars)   |    |  (4 vars)   |    | (6 vars)  |             |
|   +------+------+    +------+------+    +-----+-----+             |
|          |                  |                 |                    |
|          v                  v                 v                    |
|   +-------------+    +-------------+    +-----------+             |
|   | Investing   |    |    EMBI     |    | TRM/IBR   |             |
|   | (7 vars)    |    |  (BCRP)     |    |(Selenium) |             |
|   +------+------+    +------+------+    +-----+-----+             |
|          |                  |                 |                    |
|          +--------+---------+---------+-------+                    |
|                   |                                                |
|                   v                                                |
|          +----------------+                                        |
|          |  l0_macro_     |   Schedule: 50 12 * * 1-5             |
|          |  unified DAG   |   (7:50 AM COT, Mon-Fri)              |
|          +-------+--------+                                        |
|                  |                                                 |
|                  v                                                 |
|          +------------------+                                      |
|          |   PostgreSQL     |                                      |
|          | macro_indicators |                                      |
|          |    _daily        |                                      |
|          +--------+---------+                                      |
|                   |                                                |
+------------------------------------------------------------------+
                    |
                    | Forward Fill (bounded)
                    v
+------------------------------------------------------------------+
|                     LAYER 1 (L1) - Feature Engineering            |
|                                                                    |
|   +------------------+         +------------------+                |
|   | usdcop_m5_ohlcv  |         | macro_indicators |                |
|   | (5-min OHLCV)    |         |    _daily        |                |
|   +--------+---------+         +--------+---------+                |
|            |                            |                          |
|            +------------+---------------+                          |
|                         |                                          |
|                         v                                          |
|            +-------------------------+                             |
|            |   CanonicalFeature      |  Schedule: */5 13-17 * * 1-5|
|            |   Builder (SSOT)        |  (Every 5 min, trading hrs) |
|            |   - Wilder's EMA        |                             |
|            |   - Z-score norm        |                             |
|            |   - 15 features         |                             |
|            +------------+------------+                             |
|                         |                                          |
|                         v                                          |
|            +-------------------------+                             |
|            |  inference_features_5m  |                             |
|            |  (Feature Store)        |                             |
|            +------------+------------+                             |
|                         |                                          |
+------------------------------------------------------------------+
                          |
                          | merge_asof (backward)
                          v
+------------------------------------------------------------------+
|                     LAYER 2 (L2) - Dataset Creation               |
|                                                                    |
|            +-------------------------+                             |
|            | scripts/prepare_        |                             |
|            | training_data.py        |                             |
|            +------------+------------+                             |
|                         |                                          |
|           +-------------+-------------+                            |
|           |             |             |                            |
|           v             v             v                            |
|     +---------+   +---------+   +---------+                        |
|     | train_  |   | val_    |   | test_   |                        |
|     |features |   |features |   |features |                        |
|     |.parquet |   |.parquet |   |.parquet |                        |
|     +---------+   +---------+   +---------+                        |
|           |             |             |                            |
|           +-------------+-------------+                            |
|                         |                                          |
|                         v                                          |
|            +-------------------------+                             |
|            |       DVC Tracked       |                             |
|            |   data/processed/*.dvc  |                             |
|            +-------------------------+                             |
|                                                                    |
+------------------------------------------------------------------+
                          |
                          v
+------------------------------------------------------------------+
|                     LAYER 3 (L3) - Model Training                 |
|                                                                    |
|            +-------------------------+                             |
|            |  scripts/train_with_    |                             |
|            |  mlflow.py              |                             |
|            +------------+------------+                             |
|                         |                                          |
|                         v                                          |
|            +-------------------------+                             |
|            |   MLflow Model Registry |                             |
|            |   - Model artifacts     |                             |
|            |   - Metrics             |                             |
|            |   - Feature contract    |                             |
|            +-------------------------+                             |
|                                                                    |
+------------------------------------------------------------------+
```

### Pipeline Summary

| Layer | DAG/Script | Input | Output | Schedule |
|-------|------------|-------|--------|----------|
| L0 | `l0_macro_unified` | External APIs | `macro_indicators_daily` | Daily 7:50 AM COT |
| L1 | `l1_feature_refresh` | OHLCV + Macro | `inference_features_5m` | Every 5 min (trading hours) |
| L2 | `prepare_training_data.py` | Feature Store | Parquet files | On-demand (DVC) |
| L3 | `train_with_mlflow.py` | Parquet + norm_stats | MLflow artifacts | On-demand (DVC) |

---

## 3. Feature Order (SSOT)

The feature order is defined in `src/core/contracts/feature_contract.py` and is **immutable**. All components must import from this single source.

### Contract Reference

```
Contract ID: CTR-FEATURE-001
Version: 2.0.0
File: src/core/contracts/feature_contract.py
```

### The 15 Features (Canonical Order)

| Index | Feature Name | Type | Unit | Description | Source |
|-------|--------------|------|------|-------------|--------|
| 0 | `log_ret_5m` | Technical | z-score | 5-minute log return | L1_features |
| 1 | `log_ret_1h` | Technical | z-score | 1-hour log return | L1_features |
| 2 | `log_ret_4h` | Technical | z-score | 4-hour log return | L1_features |
| 3 | `rsi_9` | Technical | z-score | RSI with 9 periods (Wilder's EMA) | L1_features |
| 4 | `atr_pct` | Technical | z-score | ATR as percentage of price | L1_features |
| 5 | `adx_14` | Technical | z-score | ADX with 14 periods | L1_features |
| 6 | `dxy_z` | Macro | z-score | DXY index z-score | L0_macro |
| 7 | `dxy_change_1d` | Macro | z-score | DXY 1-day percentage change | L0_macro |
| 8 | `vix_z` | Macro | z-score | VIX z-score | L0_macro |
| 9 | `embi_z` | Macro | z-score | EMBI Colombia z-score | L0_macro |
| 10 | `brent_change_1d` | Macro | z-score | Brent oil 1-day change | L0_macro |
| 11 | `rate_spread` | Macro | z-score | Interest rate spread (COL - US) | L0_macro |
| 12 | `usdmxn_change_1d` | Macro | z-score | USD/MXN 1-day change | L0_macro |
| 13 | `position` | State | raw | Current position (-1, 0, 1) | trading_state |
| 14 | `time_normalized` | Time | normalized | Time of day [0, 1] | trading_state |

### Feature Order Hash

The feature order is hashed to detect any unauthorized modifications:

```python
FEATURE_ORDER_HASH = hashlib.sha256(
    ",".join(FEATURE_ORDER).encode("utf-8")
).hexdigest()[:16]
```

This hash is logged to MLflow with every training run for audit purposes.

### Clip Ranges

| Feature | Clip Min | Clip Max |
|---------|----------|----------|
| Most z-score features | -5.0 | +5.0 |
| `rsi_9` | -3.0 | +3.0 |
| `position` | -1.0 | +1.0 |
| `time_normalized` | 0.0 | 1.0 |

---

## 4. Data Sources

### 4.1 FRED API (14 Variables)

Federal Reserve Economic Data provides US macroeconomic indicators.

| Series ID | Column Name | Frequency | Description |
|-----------|-------------|-----------|-------------|
| `DTWEXBGS` | `fxrt_index_dxy_usa_d_dxy` | Daily | Dollar Index (DXY) |
| `VIXCLS` | `volt_vix_usa_d_vix` | Daily | CBOE Volatility Index |
| `DGS10` | `finc_bond_yield10y_usa_d_ust10y` | Daily | 10-Year Treasury Yield |
| `DGS2` | `finc_bond_yield2y_usa_d_dgs2` | Daily | 2-Year Treasury Yield |
| `DPRIME` | `polr_prime_rate_usa_d_prime` | Daily | Prime Rate |
| `FEDFUNDS` | `polr_fed_funds_usa_m_fedfunds` | Monthly | Federal Funds Rate |
| `CPIAUCSL` | `infl_cpi_all_usa_m_cpiaucsl` | Monthly | CPI All Items |
| `CPILFESL` | `infl_cpi_core_usa_m_cpilfesl` | Monthly | Core CPI |
| `PCEPI` | `infl_pce_usa_m_pcepi` | Monthly | PCE Price Index |
| `UNRATE` | `labr_unemployment_usa_m_unrate` | Monthly | Unemployment Rate |
| `INDPRO` | `prod_industrial_usa_m_indpro` | Monthly | Industrial Production |
| `M2SL` | `mnys_m2_supply_usa_m_m2sl` | Monthly | M2 Money Supply |
| `UMCSENT` | `sent_consumer_usa_m_umcsent` | Monthly | Consumer Sentiment |
| `GDP` | `gdpp_real_gdp_usa_q_gdp_q` | Quarterly | Real GDP |

### 4.2 TwelveData API (4 Variables)

Real-time market data for FX pairs and commodities.

| Symbol | Column Name | Description |
|--------|-------------|-------------|
| `USD/MXN` | `fxrt_spot_usdmxn_mex_d_usdmxn` | USD/MXN spot rate |
| `USD/CLP` | `fxrt_spot_usdclp_chl_d_usdclp` | USD/CLP spot rate |
| `CL` | `comm_oil_wti_glb_d_wti` | WTI Crude Oil |
| `BZ` | `comm_oil_brent_glb_d_brent` | Brent Crude Oil |

### 4.3 BanRep SUAMECA (6 Variables)

Banco de la Republica de Colombia data via Selenium scraping.

| Serie ID | Column Name | Description |
|----------|-------------|-------------|
| 241 | `finc_rate_ibr_overnight_col_d_ibr` | IBR Overnight Rate |
| 59 | `polr_policy_rate_col_d_tpm` | Monetary Policy Rate (TPM) |
| 4170 | `fxrt_reer_bilateral_col_m_itcr` | Real Exchange Rate Index |
| 4180 | `ftrd_terms_trade_col_m_tot` | Terms of Trade Index |
| 15051 | `rsbp_reserves_international_col_m_resint` | International Reserves |
| 100002 | `infl_cpi_total_col_m_ipccol` | Colombia CPI |

### 4.4 Investing.com (7 Variables)

Financial data via Cloudscraper.

| URL Pattern | Column Name | Description |
|-------------|-------------|-------------|
| `/indices/colcap-historical-data` | `eqty_index_colcap_col_d_colcap` | COLCAP Index |
| `/rates-bonds/colombia-10-year-bond-yield` | `finc_bond_yield10y_col_d_col10y` | Colombia 10Y Bond |
| `/rates-bonds/colombia-5-year-bond-yield` | `finc_bond_yield5y_col_d_col5y` | Colombia 5Y Bond |
| `/commodities/gold` | `comm_metal_gold_glb_d_gold` | Gold Price |
| `/commodities/us-coffee-c` | `comm_agri_coffee_glb_d_coffee` | Coffee Price |

### 4.5 EMBI via BCRP Peru (1 Variable)

EMBI Colombia spread from Banco Central de Reserva del Peru.

| Source | Column Name | Description |
|--------|-------------|-------------|
| BCRP Series PD04715XD | `crsk_spread_embi_col_d_embi` | EMBI Colombia Spread |

### Forward-Fill Configuration

Each indicator has a maximum forward-fill limit based on publication frequency:

| Frequency | Max FFill Days | Example |
|-----------|----------------|---------|
| Daily | 5 | DXY, VIX |
| Monthly | 45 | CPI, FEDFUNDS |
| Quarterly | 120 | GDP |

---

## 5. Temporal Joins

### The Problem

Macro data is published at different frequencies (daily, monthly, quarterly) while trading data arrives every 5 minutes. We need to join these datasets without introducing look-ahead bias.

### The Solution: `merge_asof` with `direction='backward'`

```python
import pandas as pd

# OHLCV data (5-min bars)
df_ohlcv = pd.DataFrame({
    'time': ['2026-01-15 09:00', '2026-01-15 09:05', '2026-01-15 09:10'],
    'close': [4250.0, 4252.5, 4251.0]
})

# Macro data (daily)
df_macro = pd.DataFrame({
    'date': ['2026-01-14', '2026-01-15'],
    'dxy': [103.5, 103.8],
    'vix': [18.2, 17.9]
})

# Convert to datetime
df_ohlcv['time'] = pd.to_datetime(df_ohlcv['time'])
df_ohlcv['date'] = df_ohlcv['time'].dt.normalize()
df_macro['date'] = pd.to_datetime(df_macro['date'])

# Temporal join: use most recent available macro data
result = pd.merge_asof(
    df_ohlcv.sort_values('date'),
    df_macro.sort_values('date'),
    on='date',
    direction='backward'  # CRITICAL: only look back, never forward
)
```

### Key Behavior

| OHLCV Time | Macro Date Used | Why |
|------------|-----------------|-----|
| 2026-01-15 09:00 | 2026-01-14 | 2026-01-15 macro not yet available at market open |
| 2026-01-15 13:00 | 2026-01-15 | After L0 DAG runs at 12:50 UTC (7:50 COT) |

### Why `direction='backward'`?

- **Prevents look-ahead bias**: Model only sees data that was actually available at prediction time
- **Matches production behavior**: Inference uses the same backward-looking joins
- **Audit-friendly**: Clear temporal boundaries for regulatory review

---

## 6. Trading Hours Filter

### Colombian Trading Session

The USD/COP market operates during specific hours aligned with Colombian business hours.

```
Timezone: America/Bogota (UTC-5)
Trading Hours: 08:00 - 17:00 (local time)
Trading Days: Monday - Friday (excluding Colombian holidays)
```

### UTC Conversion

| Local Time (COT) | UTC Time |
|------------------|----------|
| 08:00 | 13:00 |
| 12:00 | 17:00 |
| 17:00 | 22:00 |

### Implementation

```python
import pytz
from datetime import datetime, time

BOGOTA_TZ = pytz.timezone('America/Bogota')
MARKET_OPEN = time(8, 0)   # 08:00 local
MARKET_CLOSE = time(17, 0) # 17:00 local

def is_trading_hour(timestamp: datetime) -> bool:
    """Check if timestamp falls within Colombian trading hours."""
    local_time = timestamp.astimezone(BOGOTA_TZ)

    # Check weekday (0=Monday, 6=Sunday)
    if local_time.weekday() >= 5:
        return False

    # Check time range
    current_time = local_time.time()
    return MARKET_OPEN <= current_time <= MARKET_CLOSE
```

### DAG Schedule Alignment

| DAG | Cron Schedule | Local Time | Purpose |
|-----|---------------|------------|---------|
| L0 Macro | `50 12 * * 1-5` | 7:50 AM COT | Pre-session macro update |
| L1 Features | `*/5 13-17 * * 1-5` | 8:00 AM - 12:55 PM COT | Real-time feature refresh |

### Colombian Holidays (2026)

The trading calendar excludes Colombian national holidays:

- January 6: Epiphany
- March 23: San Jose
- April 2-3: Holy Week
- May 1: Labor Day
- May 18: Ascension
- June 8: Corpus Christi
- June 15: Sacred Heart
- June 29: Saints Peter and Paul
- July 20: Independence Day
- August 7: Battle of Boyaca
- August 17: Assumption
- October 12: Columbus Day
- November 2: All Saints
- November 16: Independence of Cartagena
- December 8: Immaculate Conception
- December 25: Christmas

---

## 7. Dataset Versioning

### DVC Configuration

The project uses DVC (Data Version Control) to track datasets and ensure reproducibility.

```yaml
# dvc.yaml stages
stages:
  prepare_data:
    cmd: python scripts/prepare_training_data.py
    deps:
      - data/raw/
      - scripts/prepare_training_data.py
      - config/feature_config.json
    outs:
      - data/processed/:
          persist: true
```

### Common DVC Commands

```bash
# Initialize DVC (first time)
dvc init

# Track a dataset
dvc add data/processed/train_features.parquet

# Push to remote storage (MinIO)
dvc push

# Pull specific dataset version
dvc pull data/processed/train_features.parquet.dvc

# Reproduce pipeline from scratch
dvc repro

# Show pipeline DAG
dvc dag

# Compare metrics between versions
dvc metrics diff

# Show data lineage
dvc dag --md
```

### Remote Storage Configuration

```bash
# Configure MinIO as DVC remote
dvc remote add -d minio s3://usdcop-dvc-store
dvc remote modify minio endpointurl http://minio:9000
dvc remote modify minio access_key_id ${MINIO_ACCESS_KEY}
dvc remote modify minio secret_access_key ${MINIO_SECRET_KEY}
```

### Dataset Versioning Best Practices

1. **Never modify tracked files directly** - Always go through the pipeline
2. **Commit `.dvc` files to Git** - They contain the hash of the actual data
3. **Tag releases** - Use Git tags to mark dataset versions used in production models
4. **Lock parameters** - Use `params.yaml` to freeze hyperparameters

```bash
# Example: Tag a dataset release
git tag -a "dataset-v1.2.0" -m "Added rate_spread feature"
dvc push
git push --tags
```

---

## 8. Normalization Statistics

### Purpose

Normalization statistics (`norm_stats.json`) contain the mean and standard deviation calculated from **training data only** to prevent data leakage.

### File Location

```
config/norm_stats.json
```

### Schema

```json
{
  "log_ret_5m": {
    "mean": 9.042127679274034e-07,
    "std": 0.0011338119633965713,
    "min": -0.049620384225421,
    "max": 0.05,
    "count": 84671,
    "null_pct": 0.0
  },
  "rsi_9": {
    "mean": 48.552482317304815,
    "std": 23.916683229459384,
    "min": 0.0,
    "max": 99.99999999946029,
    "count": 84671,
    "null_pct": 0.0
  }
}
```

### Generation Process

```python
# From dvc.yaml calculate_norm_stats stage
import pandas as pd
import json

# Load training data ONLY (never validation or test)
df = pd.read_parquet('data/processed/train_features.parquet')

features = [
    'log_ret_5m', 'log_ret_1h', 'log_ret_4h',
    'rsi_9', 'atr_pct', 'adx_14',
    'dxy_z', 'dxy_change_1d', 'vix_z', 'embi_z',
    'brent_change_1d', 'rate_spread', 'usdmxn_change_1d'
]

stats = {}
for feat in features:
    stats[feat] = {
        'mean': float(df[feat].mean()),
        'std': float(df[feat].std()),
        'min': float(df[feat].min()),
        'max': float(df[feat].max()),
        'count': int(df[feat].count()),
        'null_pct': float(df[feat].isna().mean())
    }

with open('config/norm_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
```

### Usage in Training and Inference

```python
import json
import numpy as np

def load_norm_stats():
    with open('config/norm_stats.json') as f:
        return json.load(f)

def normalize_feature(value: float, feature_name: str, stats: dict) -> float:
    """Apply z-score normalization."""
    s = stats[feature_name]
    return (value - s['mean']) / (s['std'] + 1e-8)

def clip_feature(value: float, clip_min: float = -5.0, clip_max: float = 5.0) -> float:
    """Clip to prevent extreme outliers."""
    return np.clip(value, clip_min, clip_max)
```

### JSON Schema Validation

The norm_stats file is validated against a JSON schema:

```
config/schemas/norm_stats.schema.json
```

---

## 9. Reproducibility Checklist

Every training run must log the following artifacts to MLflow for full reproducibility.

### Required MLflow Artifacts

| Artifact | Type | Purpose |
|----------|------|---------|
| `feature_contract.json` | JSON | Feature order and specs (CTR-FEATURE-001) |
| `feature_order_hash` | Tag | SHA256 hash of feature order |
| `norm_stats.json` | JSON | Normalization statistics |
| `norm_stats_hash` | Tag | SHA256 hash of norm_stats |
| `data_version` | Tag | DVC hash of training data |
| `git_commit` | Tag | Git commit SHA |
| `random_seed` | Param | Random seed for reproducibility |
| `train_start_date` | Param | First date in training set |
| `train_end_date` | Param | Last date in training set |
| `builder_version` | Tag | CanonicalFeatureBuilder version |

### Pre-Training Checklist

```bash
# 1. Verify feature contract
python -c "from src.core.contracts.feature_contract import FEATURE_ORDER_HASH; print(FEATURE_ORDER_HASH)"

# 2. Verify norm_stats exists and is valid
python -c "import json; json.load(open('config/norm_stats.json'))"

# 3. Verify DVC data is up to date
dvc status

# 4. Run DVC pipeline
dvc repro

# 5. Verify training data checksums
dvc diff
```

### MLflow Logging Example

```python
import mlflow
import hashlib
import json

def log_reproducibility_artifacts():
    """Log all artifacts required for reproducibility."""

    # Feature contract
    from src.core.contracts.feature_contract import (
        FEATURE_ORDER, FEATURE_ORDER_HASH, FEATURE_CONTRACT
    )

    mlflow.log_dict(FEATURE_CONTRACT.to_dict(), "feature_contract.json")
    mlflow.set_tag("feature_order_hash", FEATURE_ORDER_HASH)

    # Norm stats
    with open('config/norm_stats.json') as f:
        norm_stats = json.load(f)

    norm_stats_hash = hashlib.sha256(
        json.dumps(norm_stats, sort_keys=True).encode()
    ).hexdigest()[:16]

    mlflow.log_artifact('config/norm_stats.json')
    mlflow.set_tag("norm_stats_hash", norm_stats_hash)

    # DVC data version
    import subprocess
    dvc_hash = subprocess.check_output(
        ['dvc', 'get', '--show-url', 'data/processed/']
    ).decode().strip()
    mlflow.set_tag("data_version", dvc_hash)

    # Git commit
    git_commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD']
    ).decode().strip()
    mlflow.set_tag("git_commit", git_commit)
```

### Post-Training Validation

```bash
# 1. Verify model was registered
mlflow models list --name usdcop-ppo-model

# 2. Verify all artifacts logged
mlflow runs get --run-id <RUN_ID>

# 3. Compare with previous run
mlflow runs compare <RUN_ID_1> <RUN_ID_2>

# 4. Export reproducibility report
python scripts/generate_model_card.py --run-id <RUN_ID>
```

### Audit Trail Requirements

For regulatory compliance, maintain:

1. **Data Lineage**: Complete trace from raw data to model predictions
2. **Version History**: All changes to feature definitions or normalization stats
3. **Access Logs**: Who accessed/modified training data
4. **Model Provenance**: Link between deployed model and training artifacts

---

## Appendix A: Quick Reference

### File Locations

| Component | Path |
|-----------|------|
| Feature Contract | `src/core/contracts/feature_contract.py` |
| Norm Stats | `config/norm_stats.json` |
| Norm Stats Schema | `config/schemas/norm_stats.schema.json` |
| DVC Config | `dvc.yaml` |
| DVC Lock | `dvc.lock` |
| Training Data | `data/processed/*.parquet` |
| Feature Config | `config/feature_config.json` |

### Key Commands

```bash
# Full pipeline reproduction
dvc repro

# Check data status
dvc status

# Pull latest data
dvc pull

# Push data changes
dvc push

# Show pipeline graph
dvc dag

# Run specific stage
dvc repro train
```

### Contact

For questions about dataset construction:
- **Data Engineering**: data-team@company.com
- **ML Engineering**: ml-team@company.com
- **Trading Operations**: trading-ops@company.com

---

*Document generated as part of P0-04 remediation plan.*
