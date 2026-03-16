# Disaster Recovery Playbook

> **Version**: 1.0.0
> **Last Updated**: 2026-01-14
> **Owner**: Trading Team

## Overview

This playbook documents procedures for recovering the USDCOP RL Trading system from various failure scenarios.

## Contact Information

| Role | Contact |
|------|---------|
| Primary On-Call | trading-oncall@example.com |
| Database Admin | dba@example.com |
| Infrastructure | infra@example.com |

## Recovery Time Objectives

| Service | RTO | RPO |
|---------|-----|-----|
| Trading API | 15 min | 5 min |
| Database (TimescaleDB) | 30 min | 1 hour |
| Model Inference | 10 min | N/A |
| Dashboard | 60 min | 24 hours |

## Scenario 1: Database Failure

### Symptoms
- API returns 500 errors with "database connection" messages
- Airflow DAGs fail at database operations
- Dashboard shows no data

### Recovery Steps

1. **Check database status**
   ```bash
   docker ps | grep postgres
   docker logs usdcop-postgres --tail 100
   ```

2. **Restart database container**
   ```bash
   docker-compose restart postgres
   ```

3. **If corruption detected, restore from backup**
   ```bash
   # List available backups
   ls -la /backups/postgres/

   # Restore latest backup
   pg_restore -h localhost -U postgres -d usdcop_trading /backups/postgres/latest.dump
   ```

4. **Verify data integrity**
   ```bash
   psql -h localhost -U postgres -d usdcop_trading -c "SELECT COUNT(*) FROM usdcop_m5_ohlcv;"
   ```

## Scenario 2: Model Inference Failure

### Symptoms
- `/v1/health` returns unhealthy
- Inference API returns 503
- No new signals generated

### Recovery Steps

1. **Check model file integrity**
   ```bash
   md5sum models/ppo_production/final_model.zip
   # Compare with expected hash in model card
   ```

2. **Reload model**
   ```bash
   curl -X POST http://localhost:8000/v1/models/reload
   ```

3. **If model corrupted, restore from DVC**
   ```bash
   dvc pull models/ppo_production.dvc
   ```

4. **Restart inference service**
   ```bash
   docker-compose restart inference-api
   ```

## Scenario 3: Redis Failure

### Symptoms
- Real-time signals not updating
- Dashboard shows stale data
- Redis streams empty

### Recovery Steps

1. **Check Redis status**
   ```bash
   docker exec -it redis redis-cli ping
   ```

2. **Restart Redis**
   ```bash
   docker-compose restart redis
   ```

3. **Verify streams**
   ```bash
   docker exec -it redis redis-cli XLEN signals:ppo_primary:stream
   ```

## Scenario 4: Complete System Recovery

### Full Stack Restart

```bash
# Stop all services
docker-compose down

# Clean up volumes (optional, data loss!)
# docker volume prune

# Start infrastructure first
docker-compose up -d postgres redis minio

# Wait for database readiness
sleep 30

# Start application services
docker-compose up -d inference-api airflow-webserver airflow-scheduler

# Start dashboard
cd usdcop-trading-dashboard && npm run start
```

### Verification Checklist

- [ ] Database responding: `psql -h localhost -U postgres -c "SELECT 1"`
- [ ] Redis responding: `redis-cli ping`
- [ ] API healthy: `curl http://localhost:8000/v1/health`
- [ ] Airflow UI accessible: `http://localhost:8080`
- [ ] Dashboard loading: `http://localhost:3000`

## Backup Procedures

### Daily Automated Backups

Backups run automatically via cron at 02:00 UTC:

```bash
# Database backup
pg_dump -h localhost -U postgres usdcop_trading > /backups/postgres/$(date +%Y%m%d).dump

# Model backup (via DVC)
dvc push
```

### Manual Backup

```bash
# Create manual database backup
./scripts/backup_database.sh

# Push models to DVC remote
dvc push
```

## Monitoring Alerts

Configure alerting for:

1. **Database connection pool exhaustion**
2. **API latency > 1s**
3. **Model inference failures**
4. **Disk space < 20%**
5. **Memory usage > 90%**

## Rollback Procedures

### Model Rollback

```bash
# List model versions
git log --oneline models/ppo_production.dvc

# Checkout previous version
git checkout HEAD~1 -- models/ppo_production.dvc
dvc checkout models/ppo_production.dvc

# Reload model
curl -X POST http://localhost:8000/v1/models/reload
```

### Configuration Rollback

```bash
# Rollback config changes
git checkout HEAD~1 -- config/trading_config.yaml
git checkout HEAD~1 -- config/norm_stats.json
```

## Post-Recovery Validation

After any recovery:

1. Run health checks
2. Verify data freshness
3. Check signal generation
4. Monitor logs for errors
5. Update incident log
