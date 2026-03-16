# Service Level Agreements (SLA)

## Overview

This document defines the Service Level Agreements for the USDCOP trading system, including latency targets, availability requirements, and alerting thresholds.

## Inference API SLAs

### Latency Targets

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| **p50 latency** | < 20ms | > 30ms | > 50ms |
| **p95 latency** | < 50ms | > 75ms | > 100ms |
| **p99 latency** | < 100ms | > 150ms | > 200ms |

### Latency Breakdown by Stage

| Stage | p50 Target | p99 Target | Notes |
|-------|------------|------------|-------|
| Feature calculation | < 10ms | < 40ms | RSI, ATR, ADX with Wilder's EMA |
| Model inference | < 5ms | < 30ms | ONNX Runtime with CPU |
| Database query | < 5ms | < 30ms | OHLCV + macro data fetch |
| **Total** | < 20ms | < 100ms | End-to-end |

### Availability

| Metric | Target |
|--------|--------|
| Uptime | 99.9% (8.76 hours downtime/year) |
| Error rate | < 0.1% |
| Recovery time | < 5 minutes |

## Prometheus Metrics

### Latency Histograms

```promql
# Inference latency
usdcop_inference_latency_seconds

# Feature calculation time
usdcop_feature_calculation_seconds

# Database query time
usdcop_db_query_seconds
```

### Histogram Buckets

```python
LATENCY_BUCKETS = [
    0.001,   # 1ms
    0.005,   # 5ms
    0.01,    # 10ms
    0.025,   # 25ms
    0.05,    # 50ms
    0.1,     # 100ms
    0.25,    # 250ms
    0.5,     # 500ms
    1.0,     # 1s
    2.5,     # 2.5s
]
```

## Alert Rules

### Warning Alerts (Slack notification)

| Alert | Condition | Duration |
|-------|-----------|----------|
| High p95 latency | p95 > 75ms | 5 minutes |
| Elevated error rate | error_rate > 0.5% | 5 minutes |
| Feature circuit breaker | activated | immediate |

### Critical Alerts (PagerDuty)

| Alert | Condition | Duration |
|-------|-----------|----------|
| p99 latency SLA breach | p99 > 200ms | 2 minutes |
| High error rate | error_rate > 1% | 1 minute |
| Service unavailable | up == 0 | 1 minute |

## Measurement

### How Latency is Measured

```python
import time

start = time.perf_counter()
# ... operation ...
latency_ms = (time.perf_counter() - start) * 1000

# Report to Prometheus
INFERENCE_LATENCY.observe(latency_ms / 1000)  # Convert to seconds
```

### Percentile Calculation

```promql
# p50 (median)
histogram_quantile(0.5, rate(usdcop_inference_latency_seconds_bucket[5m]))

# p95
histogram_quantile(0.95, rate(usdcop_inference_latency_seconds_bucket[5m]))

# p99
histogram_quantile(0.99, rate(usdcop_inference_latency_seconds_bucket[5m]))
```

## Performance Optimization Guidelines

### Feature Calculation

1. Use batch calculations where possible
2. Cache computed features for 5-minute windows
3. Pre-compute macro features daily

### Model Inference

1. Use ONNX Runtime with graph optimization
2. Warm up model on startup (10 iterations)
3. Use appropriate thread pool size (2-4 threads)

### Database Queries

1. Use connection pooling
2. Index on timestamp columns
3. Limit query window to necessary data

## Load Testing

### Test Scenarios

| Scenario | Users | RPS | Duration |
|----------|-------|-----|----------|
| Normal load | 10 | 2 | 10 min |
| Peak load | 50 | 10 | 5 min |
| Stress test | 100 | 20 | 2 min |

### SLA Assertions in Tests

```python
def test_latency_sla():
    results = run_load_test(users=50, duration=60)

    assert results['p50'] < 20, f"p50 SLA breach: {results['p50']}ms"
    assert results['p95'] < 50, f"p95 SLA breach: {results['p95']}ms"
    assert results['p99'] < 100, f"p99 SLA breach: {results['p99']}ms"
```

## Reporting

### Daily Metrics Report

- Average latency by stage
- p50, p95, p99 percentiles
- Error rate
- Uptime percentage

### Weekly Review

- SLA compliance percentage
- Top latency contributors
- Performance trends
- Optimization opportunities

## Escalation Procedures

### Level 1 (Warning)

- Slack notification to #alerts channel
- On-call engineer investigates
- Target resolution: 30 minutes

### Level 2 (Critical)

- PagerDuty alert
- Immediate investigation required
- Target resolution: 15 minutes
- Escalate to Level 3 if unresolved

### Level 3 (Emergency)

- All-hands alert
- Consider circuit breaker activation
- Post-incident review required

## Related Documentation

- [ARCHITECTURE.md](./ARCHITECTURE.md) - System architecture
- [DATA_VERSIONING.md](./DATA_VERSIONING.md) - Data management
- `config/prometheus/alerts/` - Alert rule definitions
- `config/grafana/dashboards/` - Dashboard configurations
