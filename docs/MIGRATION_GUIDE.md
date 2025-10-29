# USDCOP Trading System - Migration Guide
## Migrating from V1.0 to V2.0 Architecture

**Version:** 2.0.0
**Date:** October 22, 2025
**Estimated Downtime:** 30-60 minutes (with zero-downtime option)

---

## Table of Contents

1. [Overview](#overview)
2. [What's Changed](#whats-changed)
3. [Pre-Migration Checklist](#pre-migration-checklist)
4. [Migration Strategy](#migration-strategy)
5. [Step-by-Step Migration](#step-by-step-migration)
6. [Post-Migration Validation](#post-migration-validation)
7. [Rollback Plan](#rollback-plan)
8. [Zero-Downtime Migration](#zero-downtime-migration)

---

## Overview

Version 2.0 introduces significant architectural improvements to the USDCOP Trading System:

- **Real-Time Orchestrator**: New service for live market data coordination
- **Separated Services**: Trading API, Analytics API, Compliance API, Pipeline API
- **Enhanced Pipeline**: Intelligent gap detection and 16 API key support
- **WebSocket v2**: Dedicated service with Redis pub/sub
- **TimescaleDB**: Hypertables for optimized time-series queries

### Compatibility

- **Database**: PostgreSQL schema changes (new columns, hypertables)
- **APIs**: Backward compatible with V1 endpoints
- **Dashboard**: No changes required (APIs compatible)
- **Docker Compose**: New services and configuration

---

## What's Changed

### Architecture Changes

| Component | V1.0 | V2.0 | Impact |
|-----------|------|------|--------|
| **RT Data Collection** | Combined with L0 pipeline | Separate RT Orchestrator service | Medium |
| **API Services** | Single Trading API | 4 specialized APIs | Low (backward compatible) |
| **WebSocket** | Embedded in Trading API | Dedicated service with Redis | Medium |
| **Database** | Standard PostgreSQL | TimescaleDB with hypertables | High |
| **L0 Pipeline** | 8 API keys | 16 API keys (2 groups) | Low |
| **Storage** | PostgreSQL only | PostgreSQL + MinIO archival | Medium |

### Breaking Changes

1. **WebSocket Endpoint**:
   - **Old**: `ws://localhost:8000/ws`
   - **New**: `ws://localhost:8082/ws/market-data`
   - **Action**: Update client WebSocket URLs

2. **Environment Variables**:
   - **New**: `API_KEY_G1_*` and `API_KEY_G2_*` (16 keys)
   - **New**: `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`
   - **Action**: Update `.env` file

3. **Database Schema**:
   - **New**: Hypertable for `usdcop_m5_ohlcv`
   - **New**: Columns: `source`, technical indicators
   - **Action**: Run migration scripts

### Non-Breaking Changes

- All V1 API endpoints still work
- Dashboard requires no code changes
- Existing data is preserved

---

## Pre-Migration Checklist

### 1. Backup Everything

```bash
# Full database backup
python scripts/backup_restore_system.py backup --output /backups/pre_migration_$(date +%Y%m%d).sql

# Backup configuration files
tar -czf /backups/config_backup_$(date +%Y%m%d).tar.gz \
  docker-compose.yml .env nginx/nginx.conf

# Verify backup
ls -lh /backups/
```

### 2. Check System Health

```bash
# Check all services are running
docker ps

# Check database connectivity
docker exec usdcop-postgres-timescale pg_isready -U admin

# Check disk space (need at least 20GB free)
df -h
```

### 3. Document Current State

```bash
# Record current data count
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c \
  "SELECT COUNT(*) AS total_records FROM usdcop_m5_ohlcv;"

# Record current API ports
docker ps --format "table {{.Names}}\t{{.Ports}}"

# Save current Docker Compose config
docker compose config > /backups/docker-compose-v1.yml
```

### 4. Schedule Maintenance Window

- **Recommended**: Outside market hours (after 1 PM COT or weekends)
- **Duration**: 30-60 minutes for standard migration
- **Duration**: 2-3 hours for zero-downtime migration

### 5. Notify Stakeholders

```
Subject: USDCOP Trading System Upgrade - Scheduled Maintenance

Dear Team,

We will be upgrading the USDCOP Trading System to V2.0 on:
Date: [DATE]
Time: [TIME] COT
Duration: 30-60 minutes
Impact: Brief service interruption

New Features:
- Real-time data orchestration
- Enhanced performance with TimescaleDB
- Improved monitoring and observability

Thank you for your patience.
```

---

## Migration Strategy

### Option A: Standard Migration (30-60 min downtime)

**Best for**: Development, staging, low-traffic environments

**Steps**:
1. Stop all services
2. Upgrade database schema
3. Update Docker Compose configuration
4. Start new services
5. Validate

### Option B: Blue-Green Migration (Zero downtime)

**Best for**: Production, high-traffic environments

**Steps**:
1. Deploy V2.0 in parallel (green)
2. Run V1.0 and V2.0 simultaneously
3. Switch traffic from V1.0 (blue) to V2.0 (green)
4. Monitor for issues
5. Decommission V1.0

---

## Step-by-Step Migration

### Standard Migration (Option A)

#### Step 1: Stop Services

```bash
cd /home/azureuser/USDCOP-RL-Models

# Stop application services (keep database running)
docker compose stop trading-api analytics-api dashboard
docker compose stop airflow-scheduler airflow-webserver

# Verify only infrastructure is running
docker ps
```

#### Step 2: Pull Latest Code

```bash
# Pull latest changes
git fetch origin
git checkout main
git pull origin main

# Verify version
git log -1 --oneline
```

#### Step 3: Update Environment Variables

```bash
# Backup old .env
cp .env .env.backup

# Copy new template
cp .env.example .env.new

# Merge configurations (keep existing passwords, add new variables)
# Edit .env.new manually to include:
# - All API_KEY_G1_* and API_KEY_G2_* variables
# - MINIO_* variables
# - New service ports

nano .env.new

# Replace .env
mv .env.new .env
```

**Required New Variables**:

```bash
# Add to .env

# RT Orchestrator
PORT_RT_ORCHESTRATOR=8085

# Enhanced L0 Pipeline API Keys (Group 1)
API_KEY_G1_1=your_key_here
API_KEY_G1_2=your_key_here
API_KEY_G1_3=your_key_here
API_KEY_G1_4=your_key_here
API_KEY_G1_5=your_key_here
API_KEY_G1_6=your_key_here
API_KEY_G1_7=your_key_here
API_KEY_G1_8=your_key_here

# Enhanced L0 Pipeline API Keys (Group 2)
API_KEY_G2_1=your_key_here
API_KEY_G2_2=your_key_here
# ... (add all 16 keys)

# MinIO
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
```

#### Step 4: Database Migration

```bash
# Run database migrations
docker exec -i usdcop-postgres-timescale psql -U admin -d usdcop_trading < migrations/001_add_timescale_hypertables.sql
docker exec -i usdcop-postgres-timescale psql -U admin -d usdcop_trading < migrations/002_add_source_column.sql
docker exec -i usdcop-postgres-timescale psql -U admin -d usdcop_trading < migrations/003_add_technical_indicators.sql

# Verify migrations
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "\d usdcop_m5_ohlcv"
```

**Migration Scripts**:

```sql
-- migrations/001_add_timescale_hypertables.sql
-- Convert to hypertable
SELECT create_hypertable('usdcop_m5_ohlcv', 'time', if_not_exists => TRUE);

-- Set chunk interval to 7 days
SELECT set_chunk_time_interval('usdcop_m5_ohlcv', INTERVAL '7 days');

-- Enable compression after 30 days
ALTER TABLE usdcop_m5_ohlcv SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'time DESC',
    timescaledb.compress_segmentby = 'symbol'
);

SELECT add_compression_policy('usdcop_m5_ohlcv', INTERVAL '30 days');

-- migrations/002_add_source_column.sql
ALTER TABLE usdcop_m5_ohlcv ADD COLUMN IF NOT EXISTS source TEXT DEFAULT 'L0';
CREATE INDEX IF NOT EXISTS idx_source ON usdcop_m5_ohlcv(source);

-- migrations/003_add_technical_indicators.sql
ALTER TABLE usdcop_m5_ohlcv ADD COLUMN IF NOT EXISTS ema_20 NUMERIC(20,8);
ALTER TABLE usdcop_m5_ohlcv ADD COLUMN IF NOT EXISTS ema_50 NUMERIC(20,8);
ALTER TABLE usdcop_m5_ohlcv ADD COLUMN IF NOT EXISTS rsi NUMERIC(10,2);
-- ... (add all indicator columns)
```

#### Step 5: Build New Docker Images

```bash
# Build all services
docker compose build --no-cache

# Verify images
docker images | grep usdcop
```

#### Step 6: Initialize MinIO

```bash
# Start MinIO and initialize buckets
docker compose up -d minio
docker compose up minio-init

# Verify buckets
docker exec usdcop-minio-init mc ls minio
```

#### Step 7: Start Services

```bash
# Start infrastructure
docker compose up -d postgres redis minio

# Start Airflow
docker compose up -d airflow-init
docker compose up -d airflow-scheduler airflow-webserver

# Start APIs
docker compose up -d trading-api analytics-api pipeline-data-api compliance-api

# Start RT Orchestrator
docker compose up -d usdcop-realtime-orchestrator

# Start WebSocket service
docker compose up -d websocket-service

# Start dashboard (last)
docker compose up -d dashboard

# Check all services
docker ps
```

#### Step 8: Validate Services

```bash
# Health checks
curl http://localhost:8000/api/health
curl http://localhost:8001/api/health
curl http://localhost:8002/api/health
curl http://localhost:8003/api/health
curl http://localhost:8085/health

# Check dashboard
curl http://localhost:5000/api/health

# Check Airflow
curl http://localhost:8080/health
```

---

## Post-Migration Validation

### 1. Data Integrity Check

```bash
# Compare record counts
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading <<EOF
-- Before migration count (from notes)
-- After migration count
SELECT COUNT(*) AS after_migration FROM usdcop_m5_ohlcv;

-- Check for gaps
SELECT
    DATE(time) AS date,
    COUNT(*) AS records
FROM usdcop_m5_ohlcv
GROUP BY DATE(time)
ORDER BY date DESC
LIMIT 7;
EOF
```

### 2. API Endpoint Tests

```bash
# Run integration tests
./scripts/test_api_endpoints.sh

# Manual tests
curl http://localhost:8000/api/latest/USDCOP
curl http://localhost:8000/api/candlesticks/USDCOP?limit=100
curl http://localhost:8001/api/analytics/rl-metrics
```

### 3. WebSocket Connection Test

```bash
# Install websocat if not present
# cargo install websocat

# Test WebSocket
websocat ws://localhost:8082/ws/market-data
# Send: {"action": "subscribe", "symbols": ["USDCOP"]}
# Should receive acknowledgment and data
```

### 4. Dashboard Smoke Test

- Open http://localhost:5000
- Verify pipeline status cards load
- Navigate to /trading
- Check if charts render
- Verify real-time data updates

### 5. Performance Validation

```bash
# Check query performance
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading <<EOF
EXPLAIN ANALYZE
SELECT * FROM usdcop_m5_ohlcv
WHERE symbol = 'USDCOP'
AND time >= NOW() - INTERVAL '1 day'
ORDER BY time DESC
LIMIT 100;
EOF

# Should use index and be fast (<50ms)
```

---

## Rollback Plan

If migration fails or critical issues are found:

### Immediate Rollback (< 10 min)

```bash
# Stop all services
docker compose down

# Restore old configuration
git checkout HEAD~1
mv .env.backup .env
mv /backups/docker-compose-v1.yml docker-compose.yml

# Restore database
python scripts/backup_restore_system.py restore \
  --input /backups/pre_migration_YYYYMMDD.sql

# Start V1 services
docker compose up -d

# Verify
curl http://localhost:8000/api/health
```

### Selective Rollback (Single Service)

```bash
# Example: Rollback only RT Orchestrator
docker compose stop usdcop-realtime-orchestrator
git checkout HEAD~1 -- services/usdcop_realtime_orchestrator.py
docker compose build usdcop-realtime-orchestrator
docker compose up -d usdcop-realtime-orchestrator
```

---

## Zero-Downtime Migration

### Prerequisites

- Load balancer or NGINX reverse proxy
- Ability to run V1 and V2 in parallel
- Double the server resources (temporarily)

### Steps

#### 1. Deploy V2 in Parallel

```bash
# Clone project to new directory
cp -r /home/azureuser/USDCOP-RL-Models /home/azureuser/USDCOP-RL-Models-v2
cd /home/azureuser/USDCOP-RL-Models-v2

# Pull latest code
git checkout main
git pull origin main

# Update .env with different ports
nano .env
# Change:
# PORT_DASHBOARD=5100 (instead of 5000)
# PORT_TRADING_API=8100 (instead of 8000)
# ... all ports +100

# Start V2 services
docker compose up -d
```

#### 2. Verify V2 Works

```bash
# Test V2 endpoints
curl http://localhost:8100/api/health
curl http://localhost:5100/api/health
```

#### 3. Configure Load Balancer

```nginx
# /etc/nginx/sites-available/trading

upstream trading_api {
    server localhost:8000 weight=100;  # V1 (100% traffic)
    server localhost:8100 weight=0;     # V2 (0% traffic)
}

upstream dashboard {
    server localhost:5000 weight=100;  # V1
    server localhost:5100 weight=0;     # V2
}

server {
    listen 80;
    server_name trading.example.com;

    location /api/ {
        proxy_pass http://trading_api;
    }

    location / {
        proxy_pass http://dashboard;
    }
}
```

#### 4. Gradual Traffic Shift

```bash
# Phase 1: 10% to V2
# Update NGINX config: V1 weight=90, V2 weight=10
sudo nginx -s reload

# Monitor for 15 minutes
docker logs usdcop-trading-api -f
docker logs usdcop-dashboard -f

# Phase 2: 50% to V2
# Update NGINX config: V1 weight=50, V2 weight=50
sudo nginx -s reload

# Monitor for 30 minutes

# Phase 3: 100% to V2
# Update NGINX config: V1 weight=0, V2 weight=100
sudo nginx -s reload
```

#### 5. Decommission V1

```bash
# After 24 hours of stable V2 operation
cd /home/azureuser/USDCOP-RL-Models
docker compose down

# Remove V1 directory (optional)
# rm -rf /home/azureuser/USDCOP-RL-Models
```

---

## Common Migration Issues

### Issue: Database migration fails

**Symptoms**: "ERROR: function create_hypertable does not exist"

**Solution**:

```bash
# Install TimescaleDB extension
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c \
  "CREATE EXTENSION IF NOT EXISTS timescaledb;"

# Re-run migrations
```

---

### Issue: RT Orchestrator stuck waiting for L0

**Symptoms**: Logs show "Waiting for L0 pipeline completion..."

**Solution**:

```bash
# Manually trigger L0 DAG
docker exec usdcop-airflow-webserver airflow dags trigger usdcop_m5__01_l0_intelligent_acquire

# Or force start RT Orchestrator (emergency only)
curl -X POST http://localhost:8085/api/force-start \
  -H "Content-Type: application/json" \
  -d '{"reason": "Migration - L0 skipped"}'
```

---

### Issue: WebSocket clients can't connect

**Symptoms**: "WebSocket connection failed"

**Solution**:

```bash
# Check WebSocket service
docker logs usdcop-websocket -f

# Check Redis
docker exec usdcop-redis redis-cli -a redis123 ping

# Restart WebSocket service
docker compose restart websocket-service
```

---

## Migration Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| **Pre-Migration** | 1-2 hours | Backup, testing, preparation |
| **Migration** | 30-60 min | Database upgrade, service restart |
| **Validation** | 1-2 hours | Testing, smoke tests |
| **Monitoring** | 24 hours | Watch for issues |
| **Finalization** | - | Remove old backups, documentation |

**Total**: 3-4 hours active work + 24 hours monitoring

---

## Post-Migration Checklist

- [ ] All services healthy (`docker ps`)
- [ ] Database record count unchanged
- [ ] API endpoints return valid data
- [ ] WebSocket streaming works
- [ ] Dashboard loads without errors
- [ ] RT Orchestrator collecting data
- [ ] Airflow DAGs running successfully
- [ ] No errors in logs
- [ ] Performance metrics acceptable
- [ ] Backup of new state created
- [ ] Team notified of successful migration
- [ ] Documentation updated

---

## Support

**Migration Issues**: Contact DevOps team at devops@trading.com
**Emergency Rollback**: Call on-call engineer: [PHONE]

**Additional Resources**:
- **Architecture**: `docs/ARCHITECTURE.md`
- **Runbook**: `docs/RUNBOOK.md`
- **Development**: `docs/DEVELOPMENT.md`
