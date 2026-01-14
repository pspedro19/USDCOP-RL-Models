# Prometheus Metrics Reference

> **Version**: 1.0.0
> **Last Updated**: 2026-01-14

## Overview

This document catalogs all Prometheus metrics exposed by the USDCOP RL Trading system.

## Metric Categories

### API Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `http_requests_total` | Counter | method, path, status | Total HTTP requests |
| `http_request_duration_seconds` | Histogram | method, path | Request latency |
| `http_requests_in_progress` | Gauge | method | Current active requests |

### Trading Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `trading_signals_total` | Counter | model_id, signal | Total signals generated |
| `trading_trades_total` | Counter | model_id, side | Total trades executed |
| `trading_pnl_total` | Counter | model_id | Cumulative PnL |
| `trading_win_rate` | Gauge | model_id, window | Rolling win rate |
| `trading_sharpe_ratio` | Gauge | model_id | Current Sharpe ratio |
| `trading_max_drawdown` | Gauge | model_id | Maximum drawdown |
| `trading_position_duration_seconds` | Histogram | model_id | Position hold time |

### Model Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `inference_latency_seconds` | Histogram | model_id | Model inference time |
| `inference_errors_total` | Counter | model_id, error_type | Inference failures |
| `trading_action_distribution` | Gauge | model_id, action | Action distribution |
| `trading_action_entropy` | Gauge | model_id | Action entropy |
| `feature_drift_score` | Gauge | feature | Feature drift score |
| `model_load_time_seconds` | Gauge | model_id | Model load duration |

### Data Pipeline Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `pipeline_dag_duration_seconds` | Histogram | dag_id | DAG execution time |
| `pipeline_task_status` | Gauge | dag_id, task_id, status | Task status |
| `pipeline_features_processed` | Counter | layer | Features processed |
| `ohlcv_rows_ingested` | Counter | source | OHLCV rows ingested |
| `macro_indicators_updated` | Counter | indicator | Macro updates |

### Infrastructure Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `database_connections_active` | Gauge | database | Active DB connections |
| `database_query_duration_seconds` | Histogram | query_type | Query latency |
| `redis_commands_total` | Counter | command | Redis commands |
| `redis_memory_bytes` | Gauge | - | Redis memory usage |

## Example Queries

### Request Rate

```promql
rate(http_requests_total[5m])
```

### 95th Percentile Latency

```promql
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
```

### Error Rate

```promql
sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))
```

### Action Distribution

```promql
trading_action_distribution{model_id="ppo_primary"}
```

### Win Rate Trend

```promql
trading_win_rate{model_id="ppo_primary", window="7d"}
```

### Model Entropy Alert

```promql
trading_action_entropy{model_id="ppo_primary"} < 0.5
```

## Instrumentation

### Python (FastAPI)

```python
from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'path', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'Request latency',
    ['method', 'path']
)
```

### Exporter Endpoint

Metrics exposed at: `http://localhost:8000/metrics`

## Recording Rules

```yaml
groups:
  - name: trading_rules
    rules:
      - record: trading:win_rate_7d
        expr: |
          sum(trading_trades_total{result="win"}) /
          sum(trading_trades_total)
      - record: trading:sharpe_ratio
        expr: |
          (avg_over_time(trading_daily_return[30d]) * 252) /
          (stddev_over_time(trading_daily_return[30d]) * sqrt(252))
```
