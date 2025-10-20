# USD/COP RL Trading Pipeline API Documentation

**Version:** 1.0.0
**Date:** October 20, 2025
**Base URL:** `http://localhost:5000/api/pipeline`

## Overview

Complete REST API for accessing all layers of the USD/COP Reinforcement Learning trading pipeline. All endpoints use **real data** from:
- **PostgreSQL/TimescaleDB**: 92,936 OHLC records (2020-01-02 to 2025-10-10)
- **MinIO**: Pipeline outputs from L0-L6 processing
- **TwelveData API**: Real-time forex market data

**No hardcoded or mock data** - all responses come from actual data sources.

---

## Quick Start

### View All Available Endpoints
```bash
GET http://localhost:5000/api/pipeline/endpoints
```

Returns complete API documentation with all available endpoints, parameters, and examples.

### Test Connection
```bash
# Test L0 raw data (PostgreSQL)
GET http://localhost:5000/api/pipeline/l0/raw-data?limit=10

# Get L0 statistics
GET http://localhost:5000/api/pipeline/l0/statistics

# View latest backtest results
GET http://localhost:5000/api/pipeline/l6/backtest-results
```

---

## Data Pipeline Architecture

```
L0: Raw OHLC Data (92,936 records from 2020-2025)
    ↓
L1: Quality Gates → 929 accepted episodes (60 bars each)
    ↓
L2: Deseasonalization + HoD Baselines
    ↓
L3: Feature Engineering → 17 features per episode
    ↓
L4: RL Dataset Creation → 557 train / 186 val / 186 test
    ↓
L5: Model Training → ONNX export + Inference profiles
    ↓
L6: Backtesting → Hedge-fund grade metrics
```

---

## Layer 0: Raw Market Data

### Get Raw OHLC Data
**Endpoint:** `GET /api/pipeline/l0/raw-data`

**Description:** Multi-source raw market data with automatic fallback (PostgreSQL → MinIO → TwelveData)

**Parameters:**
- `start_date` (string): ISO date (e.g., "2024-01-01")
- `end_date` (string): ISO date (e.g., "2024-12-31")
- `limit` (number): Max records (default: 1000, max: 10000)
- `offset` (number): Pagination offset (default: 0)
- `source` (string): `postgres` | `minio` | `twelvedata` | `all` (default: postgres)

**Example:**
```bash
GET /api/pipeline/l0/raw-data?start_date=2024-01-01&end_date=2024-12-31&limit=100&source=postgres
```

**Response:**
```json
{
  "success": true,
  "count": 100,
  "data": [
    {
      "timestamp": "2024-01-01T13:00:00Z",
      "symbol": "USDCOP",
      "close": 4012.5000,
      "bid": 4011.0000,
      "ask": 4014.0000,
      "volume": 12500,
      "source": "postgres"
    }
  ],
  "metadata": {
    "source": "postgres",
    "postgres": {
      "count": 100,
      "hasMore": true,
      "table": "market_data"
    }
  },
  "pagination": {
    "limit": 100,
    "offset": 0,
    "hasMore": true
  }
}
```

### Get L0 Statistics
**Endpoint:** `GET /api/pipeline/l0/statistics`

**Description:** Aggregate statistics on L0 data quality and completeness

**Example:**
```bash
GET /api/pipeline/l0/statistics?start_date=2024-01-01
```

**Response:**
```json
{
  "success": true,
  "statistics": {
    "overview": {
      "totalRecords": 92936,
      "dateRange": {
        "earliest": "2020-01-02T07:30:00Z",
        "latest": "2025-10-10T18:55:00Z",
        "tradingDays": 1450
      },
      "priceMetrics": {
        "min": 3800.5000,
        "max": 4250.7500,
        "avg": 4012.3456,
        "stddev": 125.6789
      }
    },
    "sources": [
      {
        "source": "twelvedata",
        "count": 92936,
        "percentage": "100.00%"
      }
    ],
    "dataQuality": {
      "avgRecordsPerDay": 64.09,
      "completeness": [...]
    }
  }
}
```

---

## Layer 1: Standardized Episodes

### Get Quality Reports
**Endpoint:** `GET /api/pipeline/l1/quality-report`

**Description:** L1 quality gate reports showing episode acceptance metrics

**Parameters:**
- `run_id` (string): Specific pipeline run ID
- `start_date` (string): Filter by date
- `end_date` (string): Filter by date

**Example:**
```bash
GET /api/pipeline/l1/quality-report
```

### List L1 Episodes
**Endpoint:** `GET /api/pipeline/l1/episodes`

**Description:** List standardized 60-bar episodes that passed quality gates

**Parameters:**
- `episode_id` (string): Get specific episode
- `limit` (number): Max episodes (default: 100, max: 1000)
- `start_date` (string): Filter by episode date

**Example:**
```bash
GET /api/pipeline/l1/episodes?limit=50
```

---

## Layer 2: Prepared Data

### Get Prepared Data
**Endpoint:** `GET /api/pipeline/l2/prepared-data`

**Description:** Deseasonalized data with HoD baselines and return series

**Parameters:**
- `episode_id` (string): Specific episode
- `limit` (number): Max episodes

**Example:**
```bash
GET /api/pipeline/l2/prepared-data?limit=50
```

**Response:**
```json
{
  "success": true,
  "count": 50,
  "preparedData": [...],
  "hodBaselines": {
    "count": 25,
    "files": ["hod_baseline_2024-01-01.parquet", ...]
  }
}
```

---

## Layer 3: Engineered Features

### Get Features
**Endpoint:** `GET /api/pipeline/l3/features`

**Description:** 17 engineered features per episode with IC compliance checks

**Parameters:**
- `episode_id` (string): Specific episode
- `limit` (number): Max episodes

**Features Include:**
- Price momentum indicators
- Volatility measures
- Volume features
- Technical indicators
- Market microstructure features

**Example:**
```bash
GET /api/pipeline/l3/features?episode_id=20240101
```

---

## Layer 4: RL-Ready Dataset

### Get RL Dataset
**Endpoint:** `GET /api/pipeline/l4/dataset`

**Description:** RL-ready dataset with train/val/test splits

**Parameters:**
- `split` (string): `train` | `val` | `test`
- `episode_id` (string): Specific episode (requires split)

**Dataset Splits:**
- Train: 557 episodes (60%)
- Validation: 186 episodes (20%)
- Test: 186 episodes (20%)
- **Total: 929 episodes**

**Example:**
```bash
GET /api/pipeline/l4/dataset?split=test
```

**Response:**
```json
{
  "success": true,
  "dataset": {
    "manifest": {...},
    "splits": {
      "test": {
        "count": 186,
        "totalSize": 1234567,
        "files": [...]
      }
    }
  },
  "summary": {
    "totalEpisodes": 929,
    "trainEpisodes": 557,
    "valEpisodes": 186,
    "testEpisodes": 186
  }
}
```

---

## Layer 5: Model Serving

### Get Models
**Endpoint:** `GET /api/pipeline/l5/models`

**Description:** Trained RL models with ONNX export and serving artifacts

**Parameters:**
- `model_id` (string): Specific model ID
- `format` (string): `onnx` | `checkpoint`

**Artifacts:**
- ONNX model files (.onnx)
- Model checkpoints
- Training metrics
- Inference latency profiles

**Example:**
```bash
GET /api/pipeline/l5/models
```

**Response:**
```json
{
  "success": true,
  "models": {
    "onnx": {
      "count": 5,
      "files": [...],
      "latest": "usdcop_rl_model_20241015.onnx"
    },
    "checkpoints": {
      "count": 12,
      "totalSizeMB": "145.67"
    },
    "metrics": {
      "count": 5,
      "files": ["training_metrics_20241015.json", ...]
    }
  }
}
```

---

## Layer 6: Backtest Results

### Get Backtest Results
**Endpoint:** `GET /api/pipeline/l6/backtest-results`

**Description:** Hedge-fund grade backtest results with comprehensive performance metrics

**Parameters:**
- `run_id` (string): Specific backtest run (defaults to latest)
- `split` (string): `test` | `val` (defaults to both)
- `metric` (string): Specific metric name (e.g., "sharpe_ratio")

**Available Metrics:**
- **Sharpe Ratio**: Risk-adjusted return
- **Sortino Ratio**: Downside risk-adjusted return
- **Calmar Ratio**: Return vs maximum drawdown
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Trade Ledger**: Complete trade history
- **Daily Returns**: Daily P&L series

**Example:**
```bash
GET /api/pipeline/l6/backtest-results?split=test
```

**Response:**
```json
{
  "success": true,
  "results": {
    "runId": "L6_20241015_abc123",
    "timestamp": "2025-10-20T14:00:00Z",
    "test": {
      "kpis": {
        "sharpe_ratio": 2.34,
        "sortino_ratio": 3.12,
        "calmar_ratio": 1.89,
        "max_drawdown": -0.0456,
        "win_rate": 0.6234,
        "profit_factor": 2.15,
        "total_trades": 1245,
        "annual_return": 0.1567
      },
      "rolling": {...},
      "manifest": {...}
    }
  }
}
```

### Get Specific Metric
```bash
GET /api/pipeline/l6/backtest-results?split=test&metric=sharpe_ratio
```

---

## Data Sources

### PostgreSQL/TimescaleDB
- **Host:** localhost:5432
- **Database:** usdcop_trading
- **Table:** market_data
- **Records:** 92,936
- **Date Range:** 2020-01-02 to 2025-10-10
- **Type:** Hypertable with automatic compression

### MinIO Object Storage
- **Endpoint:** localhost:9000
- **Console:** http://localhost:9001
- **Credentials:** minioadmin / minioadmin123

**Buckets:**
- `00-raw-usdcop-marketdata` - L0 raw data archive
- `01-l1-ds-usdcop-standardize` - L1 standardized episodes
- `02-l2-ds-usdcop-prep` - L2 prepared data
- `03-l3-ds-usdcop-features` - L3 engineered features
- `04-l4-ds-usdcop-rlready` - L4 RL-ready dataset
- `05-l5-ds-usdcop-serving` - L5 model artifacts
- `usdcop-l6-backtest` - L6 backtest results

### TwelveData API
- **Symbol:** USD/COP
- **Interval:** 5 minutes
- **Purpose:** Real-time market data
- **Usage:** Fallback when PostgreSQL/MinIO unavailable

---

## Technical Specifications

### Trading Hours
- **Market:** Colombian Forex Market
- **Hours:** 8:00 AM - 12:55 PM COT (UTC-5)
- **Days:** Monday - Friday
- **Bar Frequency:** 5 minutes
- **Episode Length:** 60 bars (5 hours of trading)

### Data Format
- **Currency Pair:** USD/COP (US Dollar / Colombian Peso)
- **Precision:** 4 decimal places
- **Timestamp Format:** ISO 8601 (UTC)
- **API Response Format:** JSON

---

## Error Handling

All endpoints return standardized error responses:

```json
{
  "success": false,
  "error": "Error description",
  "details": "Detailed error message",
  "timestamp": "2025-10-20T14:00:00Z"
}
```

**Common HTTP Status Codes:**
- `200 OK` - Success
- `400 Bad Request` - Invalid parameters
- `404 Not Found` - Resource not found (bucket/run/episode)
- `500 Internal Server Error` - Server/database error

---

## Performance Notes

1. **PostgreSQL Queries**: Optimized with TimescaleDB hypertable and indexed queries
2. **MinIO Access**: Lazy bucket loading with connection pooling
3. **Pagination**: Recommended for large datasets (use limit/offset)
4. **Caching**: API documentation endpoint cached for 1 hour
5. **Max Limits**:
   - L0 raw data: 10,000 records per request
   - Episodes: 1,000 per request

---

## Development

### Environment Variables
```bash
# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=usdcop_trading
POSTGRES_USER=admin
POSTGRES_PASSWORD=admin123

# MinIO
MINIO_ENDPOINT=localhost
MINIO_PORT=9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
```

### Testing Endpoints
```bash
# Using curl
curl -X GET "http://localhost:5000/api/pipeline/l0/raw-data?limit=10"

# Using httpie
http GET "http://localhost:5000/api/pipeline/l0/statistics"

# Using browser
http://localhost:5000/api/pipeline/endpoints
```

---

## Support & Contact

For issues, questions, or feature requests:
- **Dashboard:** http://localhost:5000
- **API Base:** http://localhost:5000/api/pipeline
- **MinIO Console:** http://localhost:9001

---

## Version History

### v1.0.0 (October 20, 2025)
- Initial release
- Complete L0-L6 pipeline API endpoints
- Multi-source data support (PostgreSQL, MinIO, TwelveData)
- Real data integration (no mocks)
- Comprehensive documentation
- Hedge-fund grade metrics

---

**Note:** All endpoints are production-ready and use real data from the trading pipeline. No hardcoded or mock data is returned.
