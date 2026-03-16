# USDCOP Trading System - Operations Runbook

**Version:** 2.0.0
**Date:** October 22, 2025
**Owner:** DevOps/SRE Team

---

## Table of Contents

1. [Deployment Procedures](#deployment-procedures)
2. [Rollback Procedures](#rollback-procedures)
3. [Monitoring & Alerting](#monitoring--alerting)
4. [Incident Response](#incident-response)
5. [Backup & Recovery](#backup--recovery)
6. [Maintenance Procedures](#maintenance-procedures)
7. [Common Issues & Solutions](#common-issues--solutions)

---

## Deployment Procedures

### Standard Deployment (Zero Downtime)

**Prerequisites:**
- Access to production server
- Docker & Docker Compose installed
- Git repository access
- Backup completed (see Backup section)

**Steps:**

```bash
# 1. Connect to production server
ssh user@production-server

# 2. Navigate to project directory
cd /home/azureuser/USDCOP-RL-Models

# 3. Pull latest changes
git fetch origin
git checkout main
git pull origin main

# 4. Check for configuration changes
git diff HEAD~1 docker-compose.yml
git diff HEAD~1 .env.example

# 5. Update environment variables if needed
nano .env
# Compare with .env.example for new variables

# 6. Build new images (without stopping services)
docker compose build --no-cache

# 7. Deploy services one by one (zero downtime)

# Step 7a: Deploy backend APIs first
docker compose up -d trading-api
docker compose up -d analytics-api
docker compose up -d pipeline-data-api
docker compose up -d compliance-api

# Wait 30 seconds for health checks
sleep 30

# Step 7b: Check API health
curl http://localhost:8000/api/health
curl http://localhost:8001/api/health
curl http://localhost:8002/api/health
curl http://localhost:8003/api/health

# Step 7c: Deploy RT Orchestrator
docker compose up -d usdcop-realtime-orchestrator

# Wait for L0 dependency check (max 30 min)
docker logs usdcop-realtime-orchestrator -f --tail 50

# Step 7d: Deploy dashboard (last to minimize user impact)
docker compose up -d dashboard

# Wait 60 seconds for Next.js build
sleep 60

# 8. Verify all services are healthy
docker ps | grep usdcop

# 9. Run smoke tests
./scripts/smoke_tests.sh

# 10. Check logs for errors
docker compose logs --tail=100 | grep -i error
```

**Rollback Trigger:**
If smoke tests fail or critical errors appear in logs, immediately execute rollback procedure (see next section).

---

### Blue-Green Deployment (Maximum Safety)

**Setup:**

```bash
# Create blue and green stacks
cd /home/azureuser

# Blue (current production)
cp -r USDCOP-RL-Models USDCOP-RL-Models-blue
cd USDCOP-RL-Models-blue
# Update docker-compose.yml to use different ports (e.g., 5000 -> 5100)

# Green (new version)
cd /home/azureuser/USDCOP-RL-Models
git pull origin main
docker compose build

# Start green stack
docker compose up -d

# Test green stack
curl http://localhost:5000/api/health

# Switch traffic (update NGINX or load balancer)
# Update NGINX upstream to point to green stack

# Monitor for 30 minutes
# If stable, shut down blue stack
cd /home/azureuser/USDCOP-RL-Models-blue
docker compose down
```

---

## Rollback Procedures

### Immediate Rollback (Critical Failure)

**When to Rollback:**
- Dashboard not loading (500 errors)
- API health checks failing
- Database connection errors
- Data corruption detected
- Security vulnerability discovered

**Steps:**

```bash
# 1. Stop current services
docker compose down

# 2. Checkout previous stable version
git log --oneline -10  # Find last stable commit
git checkout <previous-commit-hash>

# 3. Restore environment variables
cp .env.backup .env

# 4. Rebuild and start
docker compose build --no-cache
docker compose up -d

# 5. Verify rollback successful
curl http://localhost:8000/api/health
curl http://localhost:5000

# 6. Restore database if needed (see Backup & Recovery)
python scripts/backup_restore_system.py restore --input /backups/latest.sql

# 7. Notify team
# Send message to Slack/Teams/Email

# 8. Create incident report
# Document what went wrong and steps taken
```

**Expected Downtime:** 5-10 minutes

---

### Partial Rollback (Single Service)

If only one service is failing:

```bash
# Example: Rollback only the dashboard
docker compose stop dashboard

# Rebuild previous version
git checkout HEAD~1 -- usdcop-trading-dashboard/
docker compose build dashboard
docker compose up -d dashboard

# Verify
curl http://localhost:5000/api/health
```

---

## Monitoring & Alerting

### Health Check Endpoints

Monitor these endpoints every 60 seconds:

| Service | Endpoint | Expected Response | Timeout |
|---------|----------|-------------------|---------|
| Dashboard | http://localhost:5000/api/health | 200 OK | 5s |
| Trading API | http://localhost:8000/api/health | 200 OK | 3s |
| Analytics API | http://localhost:8001/api/health | 200 OK | 3s |
| Pipeline API | http://localhost:8002/api/health | 200 OK | 3s |
| Compliance API | http://localhost:8003/api/health | 200 OK | 3s |
| RT Orchestrator | http://localhost:8085/health | 200 OK | 3s |
| WebSocket | ws://localhost:8082/ws | Connection OK | 5s |
| Airflow | http://localhost:8080/health | 200 OK | 5s |
| PostgreSQL | `pg_isready` | accepting connections | 2s |
| Redis | `redis-cli ping` | PONG | 1s |
| MinIO | http://localhost:9000/minio/health/live | 200 OK | 3s |

**Monitoring Script:**

```bash
#!/bin/bash
# /home/azureuser/scripts/health_check.sh

check_service() {
    local name=$1
    local url=$2
    local timeout=$3

    response=$(curl -s -o /dev/null -w "%{http_code}" -m $timeout $url)

    if [ "$response" = "200" ]; then
        echo "[OK] $name is healthy"
        return 0
    else
        echo "[FAIL] $name returned $response"
        # Send alert
        send_alert "$name is down! HTTP $response"
        return 1
    fi
}

check_service "Trading API" "http://localhost:8000/api/health" 3
check_service "Analytics API" "http://localhost:8001/api/health" 3
check_service "Dashboard" "http://localhost:5000/api/health" 5
# ... add more services

# Check database
docker exec usdcop-postgres-timescale pg_isready -U admin
if [ $? -ne 0 ]; then
    echo "[FAIL] PostgreSQL is down"
    send_alert "PostgreSQL is down!"
fi
```

**Schedule with Cron:**

```bash
# Edit crontab
crontab -e

# Add monitoring job (every 1 minute)
* * * * * /home/azureuser/scripts/health_check.sh >> /var/log/usdcop-health.log 2>&1
```

---

### Metrics to Monitor (Prometheus)

**System Metrics:**
- CPU usage per container (threshold: >80%)
- Memory usage per container (threshold: >85%)
- Disk usage (threshold: >90%)
- Network I/O

**Application Metrics:**
- API request rate (requests/sec)
- API latency (p50, p95, p99)
- Error rate (errors/min, threshold: >10)
- Database connection pool usage

**Business Metrics:**
- Market data gaps (threshold: >5 missing bars/day)
- RT Orchestrator uptime during market hours (threshold: <99%)
- Pipeline success rate (threshold: <95%)
- WebSocket client count

**Prometheus Queries:**

```promql
# API error rate
rate(http_requests_total{status=~"5.."}[5m])

# Database connection pool exhaustion
pg_stat_database_numbackends / pg_settings_max_connections > 0.9

# Pipeline failure rate
rate(pipeline_failures_total[1h]) > 0

# Disk usage alert
(node_filesystem_size_bytes - node_filesystem_free_bytes) / node_filesystem_size_bytes > 0.9
```

---

### Alerting Rules (Prometheus)

**File:** `prometheus/alerts.yml`

```yaml
groups:
  - name: usdcop_alerts
    interval: 60s
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 10
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "{{ $value }} errors/sec for {{ $labels.job }}"

      # Service down
      - alert: ServiceDown
        expr: up == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.job }} is down"
          description: "{{ $labels.instance }} has been down for >2 minutes"

      # Database connection pool exhaustion
      - alert: DatabasePoolExhausted
        expr: (pg_stat_database_numbackends / pg_settings_max_connections) > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Database connection pool nearly exhausted"
          description: "{{ $value }}% connections in use"

      # Disk space low
      - alert: DiskSpaceLow
        expr: (node_filesystem_avail_bytes / node_filesystem_size_bytes) < 0.1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Disk space low"
          description: "Only {{ $value }}% available on {{ $labels.mountpoint }}"

      # Pipeline failure
      - alert: PipelineFailure
        expr: pipeline_status{status="failed"} > 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Pipeline {{ $labels.pipeline_name }} failed"
          description: "Check Airflow logs for details"
```

---

## Incident Response

### Incident Severity Levels

| Level | Description | Response Time | Escalation |
|-------|-------------|---------------|------------|
| **P0 - Critical** | Complete system down, data loss | Immediate (24/7) | CTO, CEO |
| **P1 - High** | Major feature broken, affects >50% users | 15 minutes | Engineering Lead |
| **P2 - Medium** | Partial feature broken, affects <50% users | 1 hour | On-call engineer |
| **P3 - Low** | Minor bug, no user impact | Next business day | Support ticket |

---

### P0: Critical Incident Response

**Scenario:** Complete system outage, dashboard not loading, all APIs down

**Response:**

```bash
# 1. Acknowledge incident (0-2 min)
# Update status page, notify stakeholders

# 2. Initial triage (2-5 min)
# Check all services
docker ps -a

# Check logs
docker compose logs --tail=100 | grep -i error

# Check system resources
df -h
free -h
top

# 3. Immediate mitigation (5-15 min)
# If database is down
docker compose restart postgres
# Wait 30 seconds
docker exec usdcop-postgres-timescale pg_isready -U admin

# If all services crashed (out of memory)
# Restart services in order
docker compose up -d postgres redis minio
sleep 30
docker compose up -d trading-api analytics-api
sleep 30
docker compose up -d dashboard

# 4. Restore from backup if needed (15-30 min)
python scripts/backup_restore_system.py restore --input /backups/latest.sql

# 5. Verify system recovery
./scripts/smoke_tests.sh

# 6. Post-incident (30-60 min)
# Create incident report
# Document root cause
# Plan preventive measures
```

**Escalation Path:**
1. On-call engineer (immediate)
2. Engineering Lead (after 15 min if unresolved)
3. CTO (after 30 min if unresolved)
4. CEO (if data loss or >2 hour outage)

---

### P1: High Severity Incident

**Scenario:** RT Orchestrator not collecting data during market hours

**Response:**

```bash
# 1. Check RT Orchestrator status
docker logs usdcop-realtime-orchestrator -f --tail 100

# 2. Common causes and fixes:

# Cause A: Waiting for L0 pipeline
# Check if L0 pipeline completed
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c \
  "SELECT * FROM pipeline_status WHERE pipeline_name LIKE '%L0%' ORDER BY started_at DESC LIMIT 5;"

# If L0 stuck, restart Airflow scheduler
docker compose restart airflow-scheduler

# Cause B: API key exhausted
# Check Redis for API key status
docker exec usdcop-redis redis-cli -a redis123 get "api_key_status"

# Rotate to next API key group
docker exec usdcop-realtime-orchestrator \
  curl -X POST http://localhost:8085/api/rotate-api-keys

# Cause C: TwelveData API down
# Check external API status
curl "https://api.twelvedata.com/time_series?symbol=USDCOP&interval=5min&apikey=YOUR_KEY"

# Fallback to historical data mode
docker exec usdcop-realtime-orchestrator \
  curl -X POST http://localhost:8085/api/fallback-mode

# 3. Manual data collection if needed
python scripts/manual_data_fetch.py --symbol USDCOP --start "2025-10-22 08:00" --end "2025-10-22 13:00"
```

---

### P2: Medium Severity Incident

**Scenario:** Dashboard showing stale data (>10 minutes old)

**Response:**

```bash
# 1. Check WebSocket service
docker logs usdcop-websocket -f --tail 50

# 2. Test WebSocket connection
# Install websocat if not present
curl http://localhost:8082/health

# 3. Check Redis pub/sub
docker exec usdcop-redis redis-cli -a redis123 pubsub channels
docker exec usdcop-redis redis-cli -a redis123 pubsub numsub market_data_channel

# 4. Restart WebSocket service
docker compose restart websocket-service

# 5. Verify dashboard updates
# Open dashboard in browser, check timestamp of latest data
```

---

## Backup & Recovery

### Automated Daily Backups

**Backup Script:** `scripts/backup_restore_system.py`

**Setup Cron Job:**

```bash
# Create backup directory
sudo mkdir -p /backups/usdcop
sudo chown $USER:$USER /backups/usdcop

# Edit crontab
crontab -e

# Add backup job (daily at 2 AM)
0 2 * * * python /home/azureuser/USDCOP-RL-Models/scripts/backup_restore_system.py backup --output /backups/usdcop/backup_$(date +\%Y\%m\%d_\%H\%M\%S).sql >> /var/log/usdcop-backup.log 2>&1

# Add backup rotation (keep last 30 days)
0 3 * * * find /backups/usdcop -name "backup_*.sql" -mtime +30 -delete
```

---

### Manual Backup

```bash
# Full PostgreSQL backup
python scripts/backup_restore_system.py backup --output /backups/manual_backup_$(date +%Y%m%d).sql

# Backup specific table
docker exec usdcop-postgres-timescale pg_dump -U admin -d usdcop_trading -t usdcop_m5_ohlcv > /backups/ohlcv_backup.sql

# Backup MinIO buckets
mc mirror minio/00-raw-usdcop-marketdata /backups/minio/00-raw-usdcop-marketdata

# Backup configuration files
tar -czf /backups/config_$(date +%Y%m%d).tar.gz \
  docker-compose.yml \
  .env \
  nginx/nginx.conf \
  prometheus/prometheus.yml
```

---

### Restore from Backup

**Full System Restore:**

```bash
# 1. Stop all services (except database)
docker compose stop trading-api analytics-api dashboard

# 2. Restore PostgreSQL
python scripts/backup_restore_system.py restore --input /backups/backup_20251022_020000.sql

# 3. Verify data
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c \
  "SELECT COUNT(*) FROM usdcop_m5_ohlcv;"

# 4. Restart services
docker compose up -d

# 5. Verify system health
./scripts/smoke_tests.sh
```

**Point-in-Time Recovery (TimescaleDB):**

```bash
# Restore to specific date
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading <<EOF
-- Delete data after specific timestamp
DELETE FROM usdcop_m5_ohlcv WHERE time > '2025-10-22 10:00:00';

-- Verify
SELECT MAX(time) FROM usdcop_m5_ohlcv;
EOF
```

---

### Disaster Recovery Plan

**RTO (Recovery Time Objective):** 4 hours
**RPO (Recovery Point Objective):** 24 hours (daily backups)

**Disaster Scenarios:**

1. **Complete Server Failure**
   - Provision new server
   - Install Docker, Docker Compose
   - Clone repository
   - Restore latest backup
   - Update DNS to new server

2. **Database Corruption**
   - Stop all write services
   - Restore from last known good backup
   - Validate data integrity
   - Resume services

3. **Data Center Outage**
   - Failover to backup region (if configured)
   - Restore from S3/offsite backup
   - Update load balancer

**Disaster Recovery Checklist:**
- [ ] Access to offsite backups confirmed
- [ ] Backup restoration tested monthly
- [ ] Emergency contact list updated
- [ ] Runbook reviewed quarterly
- [ ] DR drill performed annually

---

## Maintenance Procedures

### Weekly Maintenance (Sunday 2 AM COT)

```bash
#!/bin/bash
# /home/azureuser/scripts/weekly_maintenance.sh

# 1. Database maintenance
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading <<EOF
-- Vacuum and analyze
VACUUM ANALYZE;

-- Reindex (improves query performance)
REINDEX DATABASE usdcop_trading;

-- Check for bloat
SELECT schemaname, tablename,
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
LIMIT 10;
EOF

# 2. Docker cleanup
docker system prune -af --volumes
docker volume prune -f

# 3. Log rotation
find /var/log -name "usdcop-*.log" -mtime +7 -delete

# 4. MinIO bucket cleanup (old manifests)
docker exec usdcop-minio-init mc rm --recursive --force --older-than 90d \
  minio/99-common-trading-backups

# 5. Update system packages
sudo apt-get update
sudo apt-get upgrade -y

# 6. Restart services (rolling restart)
docker compose restart trading-api
sleep 30
docker compose restart analytics-api
sleep 30
docker compose restart dashboard
```

**Schedule:**

```bash
crontab -e

# Weekly maintenance (Sunday 2 AM)
0 2 * * 0 /home/azureuser/scripts/weekly_maintenance.sh >> /var/log/usdcop-maintenance.log 2>&1
```

---

### Monthly Maintenance

**Tasks:**
1. Review and archive old logs
2. Update dependencies (npm, pip)
3. Security patches
4. Performance tuning
5. Backup validation (test restore)

```bash
# 1. Review disk usage
df -h
du -sh /var/lib/docker/volumes/*

# 2. Archive old MinIO data (>6 months)
docker exec usdcop-minio-init mc cp --recursive \
  minio/00-raw-usdcop-marketdata/USDCOP/M5/2025/04/ \
  s3://long-term-archive/

# 3. Update Docker images
docker compose pull
docker compose up -d

# 4. Test backup restore (on staging)
python scripts/backup_restore_system.py restore \
  --input /backups/latest.sql \
  --target staging_database

# 5. Generate monthly report
python scripts/generate_monthly_report.py --month 2025-10
```

---

## Common Issues & Solutions

### Issue: "Database connection pool exhausted"

**Symptoms:**
- API returns 500 errors
- Logs show "ConnectionPoolError"

**Solution:**

```bash
# 1. Check current connections
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c \
  "SELECT count(*) FROM pg_stat_activity;"

# 2. Kill idle connections
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c \
  "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle' AND state_change < NOW() - INTERVAL '5 minutes';"

# 3. Increase max connections (if needed)
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c \
  "ALTER SYSTEM SET max_connections = 200;"

# Restart PostgreSQL
docker compose restart postgres

# 4. Tune API connection pools (reduce pool size)
# Edit services/trading_api_realtime.py
# pool = await asyncpg.create_pool(max_size=10)  # Reduce from 20 to 10
```

**Prevention:**
- Monitor connection pool usage
- Set aggressive connection timeout
- Use read replicas for read-heavy queries

---

### Issue: "MinIO out of disk space"

**Symptoms:**
- Pipeline DAGs failing
- MinIO health check fails
- Error: "No space left on device"

**Solution:**

```bash
# 1. Check disk usage
df -h | grep minio

# 2. Clear old data (>6 months)
docker exec usdcop-minio-init mc rm --recursive --force --older-than 180d \
  minio/00-raw-usdcop-marketdata

# 3. Compress old buckets
docker exec usdcop-minio-init mc cp --recursive \
  minio/01-l1-ds-usdcop-standardize/ \
  minio/99-common-trading-backups/archived/

# 4. Expand volume (if on cloud)
# AWS EBS: Modify volume size
# Azure Disk: Increase disk size
# Then resize filesystem
sudo resize2fs /dev/sdb

# 5. Enable MinIO lifecycle policies
docker exec usdcop-minio-init mc ilm rule add \
  --expire-days 365 \
  minio/00-raw-usdcop-marketdata
```

---

### Issue: "RT Orchestrator stuck waiting for L0"

**Symptoms:**
- RT Orchestrator logs show "Waiting for L0 pipeline..."
- No real-time data after market opens

**Solution:**

```bash
# 1. Check L0 pipeline status
docker logs usdcop-airflow-scheduler | grep -i "l0"

# 2. Manually trigger L0 DAG
docker exec usdcop-airflow-webserver airflow dags trigger usdcop_m5__01_l0_intelligent_acquire

# 3. If L0 fails, check API keys
# Verify API key validity
curl "https://api.twelvedata.com/time_series?symbol=USDCOP&interval=5min&apikey=YOUR_KEY"

# 4. Force RT Orchestrator to skip L0 wait (emergency only)
docker exec usdcop-realtime-orchestrator curl -X POST http://localhost:8085/api/force-start

# 5. Check pipeline_status table
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c \
  "UPDATE pipeline_status SET status = 'completed' WHERE pipeline_name LIKE '%L0%' AND DATE(started_at) = CURRENT_DATE;"
```

---

### Issue: "Dashboard showing blank charts"

**Symptoms:**
- Dashboard loads but charts are empty
- No errors in console

**Solution:**

```bash
# 1. Check if data exists
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c \
  "SELECT COUNT(*), MAX(time) FROM usdcop_m5_ohlcv WHERE symbol = 'USDCOP';"

# 2. Test API endpoint
curl http://localhost:8000/api/candlesticks/USDCOP?timeframe=5m&limit=100

# 3. Check browser console for CORS errors
# If CORS error, add origin to API CORS settings

# 4. Clear browser cache
# Chrome: Ctrl+Shift+Delete
# Or hard refresh: Ctrl+Shift+R

# 5. Verify WebSocket connection
# Open browser console and check Network tab for WS connection
```

---

### Issue: "Airflow webserver won't start"

**Symptoms:**
- Airflow UI not accessible
- Container keeps restarting

**Solution:**

```bash
# 1. Check logs
docker logs usdcop-airflow-webserver --tail 100

# 2. Common cause: Database migration needed
docker exec usdcop-airflow-webserver airflow db upgrade

# 3. Reset Airflow database (CAUTION: loses task history)
docker compose down airflow-webserver airflow-scheduler
docker volume rm usdcop-rl-models_airflow_logs
docker compose up -d airflow-init
# Wait for init to complete
docker compose up -d airflow-scheduler airflow-webserver

# 4. Check port conflict
sudo lsof -i :8080
# If another service using port, change Airflow port in docker-compose.yml

# 5. Recreate admin user
docker exec usdcop-airflow-webserver airflow users create \
  --username admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@trading.com \
  --password admin123
```

---

## Escalation Contacts

| Role | Name | Phone | Email | Hours |
|------|------|-------|-------|-------|
| On-Call Engineer | [Name] | [Phone] | [Email] | 24/7 |
| Engineering Lead | [Name] | [Phone] | [Email] | 9 AM - 6 PM COT |
| DevOps Lead | [Name] | [Phone] | [Email] | 9 AM - 6 PM COT |
| Database Admin | [Name] | [Phone] | [Email] | On-call (P0 only) |
| CTO | [Name] | [Phone] | [Email] | Emergency only |

---

## Additional Resources

- **Architecture Documentation:** `docs/ARCHITECTURE.md`
- **Development Guide:** `docs/DEVELOPMENT.md`
- **API Reference:** `docs/API_REFERENCE_V2.md`
- **Migration Guide:** `docs/MIGRATION_GUIDE.md`
- **Grafana Dashboards:** http://localhost:3002
- **Prometheus Metrics:** http://localhost:9090
- **Airflow UI:** http://localhost:8080

---

## Document Updates

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-10-22 | 2.0.0 | Initial production runbook | DevOps Team |

---

**Next Review Date:** January 22, 2026

**Feedback:** Send comments/suggestions to devops@trading.com
