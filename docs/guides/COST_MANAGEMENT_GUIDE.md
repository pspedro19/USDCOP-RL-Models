# Cost Management Guide

> **Version**: 1.0.0
> **Last Updated**: 2026-01-14

## Overview

This guide documents resource allocation and cost estimation for the USDCOP RL Trading system.

## Infrastructure Costs

### Development Environment

| Resource | Specification | Monthly Cost |
|----------|---------------|--------------|
| PostgreSQL | 2 vCPU, 8GB RAM | $50 |
| Redis | 1 vCPU, 2GB RAM | $20 |
| Airflow | 2 vCPU, 4GB RAM | $40 |
| Inference API | 2 vCPU, 4GB RAM | $40 |
| MinIO | 1 vCPU, 4GB RAM, 100GB | $30 |
| **Total** | | **$180/month** |

### Production Environment (Estimated)

| Resource | Specification | Monthly Cost |
|----------|---------------|--------------|
| PostgreSQL (TimescaleDB) | 4 vCPU, 16GB RAM, 500GB SSD | $200 |
| Redis Cluster | 3 nodes, 2GB each | $90 |
| Airflow (managed) | 2 workers | $150 |
| Inference API (HA) | 2 instances, 4 vCPU each | $160 |
| Load Balancer | | $20 |
| Monitoring (Grafana Cloud) | | $50 |
| MinIO/S3 | 1TB storage | $25 |
| **Total** | | **$695/month** |

## Compute Optimization

### Model Training

| Resource | Cost | Duration | Total |
|----------|------|----------|-------|
| GPU Instance (training) | $1.50/hr | 4 hours | $6/run |
| CPU Instance (backtest) | $0.20/hr | 1 hour | $0.20/run |

### Recommended Schedule

| Task | Frequency | Monthly Cost |
|------|-----------|--------------|
| Model Retraining | Weekly | $24 |
| Full Backtests | Daily | $6 |
| Feature Refresh | Hourly | Included |

## Data Storage

### Database Growth

| Table | Growth Rate | Monthly Storage |
|-------|-------------|-----------------|
| `usdcop_m5_ohlcv` | 60 bars/day × 20 days | ~1.2K rows/month |
| `model_inferences` | 60/day × 20 days | ~1.2K rows/month |
| `macro_indicators_daily` | 20 rows/month | 20 rows/month |

**Storage Estimate**: ~5GB/year at current rate

### Retention Policy

| Data Type | Retention | Archive |
|-----------|-----------|---------|
| Raw OHLCV | 5 years | Cold storage |
| Inferences | 2 years | Archive |
| Logs | 90 days | Delete |

## Cost Reduction Strategies

### 1. Reserved Instances

- Commit to 1-year reserved for 30% savings
- Production estimate: $695 → $486/month

### 2. Spot Instances for Training

- Use spot/preemptible for non-critical workloads
- Training cost: $6 → $2/run

### 3. Data Compression

- Enable TimescaleDB compression
- Estimated 80% storage reduction

### 4. Right-sizing

- Monitor resource utilization
- Scale down underutilized services

## Budget Alerts

Configure alerts at:
- 50% of monthly budget
- 75% of monthly budget
- 90% of monthly budget

## Cost Tracking

### Monthly Review

1. Review cloud billing dashboard
2. Compare to budget
3. Identify optimization opportunities
4. Document in cost tracking spreadsheet

### Metrics to Monitor

| Metric | Target |
|--------|--------|
| Cost per inference | < $0.001 |
| Cost per backtest | < $0.50 |
| Storage cost/GB | < $0.10 |

## Scaling Considerations

### Horizontal Scaling

| Trigger | Action | Cost Impact |
|---------|--------|-------------|
| API latency > 500ms | Add inference instance | +$80/month |
| Database connections > 80% | Add read replica | +$100/month |
| Queue depth > 100 | Add Airflow worker | +$75/month |

### Vertical Scaling

| Trigger | Action | Cost Impact |
|---------|--------|-------------|
| Memory usage > 80% | Upgrade instance | +20-50% |
| CPU usage > 70% sustained | Upgrade instance | +20-50% |
