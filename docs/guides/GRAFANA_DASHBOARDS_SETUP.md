# Grafana Dashboards Setup Guide

> **Version**: 1.0.0
> **Last Updated**: 2026-01-14

## Overview

This guide covers setting up Grafana dashboards for monitoring the USDCOP RL Trading system.

## Prerequisites

- Grafana 9.0+
- Prometheus (for metrics)
- PostgreSQL (for annotations)
- Running trading system

## Access

| Environment | URL | Credentials |
|-------------|-----|-------------|
| Development | http://localhost:3001 | admin/admin |
| Production | TBD | Vault-managed |

## Data Sources Setup

### 1. Prometheus

```yaml
Name: prometheus
Type: Prometheus
URL: http://prometheus:9090
Access: Server
```

### 2. PostgreSQL

```yaml
Name: postgres
Type: PostgreSQL
Host: postgres:5432
Database: usdcop_trading
User: grafana_readonly
TLS: require
```

## Dashboard Templates

### Trading Performance Dashboard

**Panels:**

1. **Equity Curve**
   - Type: Time Series
   - Query: `SELECT timestamp_utc, cumulative_pnl FROM dw.fact_strategy_signals`

2. **Signal Distribution**
   - Type: Pie Chart
   - Query: `SELECT signal, COUNT(*) FROM trading.model_inferences GROUP BY signal`

3. **Win Rate (7-day rolling)**
   - Type: Stat
   - Query: Prometheus `trading_win_rate{window="7d"}`

4. **Sharpe Ratio**
   - Type: Gauge
   - Query: Prometheus `trading_sharpe_ratio`

### System Health Dashboard

**Panels:**

1. **API Latency**
   - Type: Time Series
   - Query: `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))`

2. **Request Rate**
   - Type: Time Series
   - Query: `rate(http_requests_total[5m])`

3. **Error Rate**
   - Type: Time Series
   - Query: `rate(http_requests_total{status=~"5.."}[5m])`

4. **Database Connections**
   - Type: Gauge
   - Query: `pg_stat_activity_count`

### Model Monitoring Dashboard

**Panels:**

1. **Action Distribution**
   - Type: Time Series (stacked)
   - Query: `trading_action_distribution{action=~"LONG|SHORT|HOLD"}`

2. **Feature Drift**
   - Type: Heatmap
   - Query: `feature_drift_score`

3. **Inference Latency**
   - Type: Histogram
   - Query: `inference_latency_seconds_bucket`

4. **Action Entropy**
   - Type: Gauge
   - Query: `trading_action_entropy`
   - Thresholds: 0.5 (red), 1.0 (yellow), 1.5 (green)

## Alert Rules

### Critical Alerts

```yaml
- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "High error rate detected"

- alert: ModelCollapse
  expr: trading_action_entropy < 0.5
  for: 30m
  labels:
    severity: critical
  annotations:
    summary: "Potential model mode collapse"

- alert: DatabaseDown
  expr: pg_up == 0
  for: 1m
  labels:
    severity: critical
```

### Warning Alerts

```yaml
- alert: HighLatency
  expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
  for: 10m
  labels:
    severity: warning

- alert: LowWinRate
  expr: trading_win_rate{window="7d"} < 0.3
  for: 24h
  labels:
    severity: warning
```

## Import/Export

### Export Dashboard

```bash
curl -H "Authorization: Bearer $GRAFANA_TOKEN" \
  http://localhost:3001/api/dashboards/uid/trading-performance \
  > dashboards/trading-performance.json
```

### Import Dashboard

```bash
curl -X POST -H "Authorization: Bearer $GRAFANA_TOKEN" \
  -H "Content-Type: application/json" \
  -d @dashboards/trading-performance.json \
  http://localhost:3001/api/dashboards/db
```

## Provisioning

Store dashboard definitions in:
```
grafana/
  provisioning/
    dashboards/
      trading-performance.json
      system-health.json
      model-monitoring.json
    datasources/
      prometheus.yml
      postgres.yml
```
