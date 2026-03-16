# Troubleshooting Guide
## USD/COP RL Trading System

**Version:** 1.0.0
**Date:** 2026-01-17

---

## Quick Diagnostics

### System Health Check

```bash
# Check all services status
docker-compose ps

# Check container logs
docker-compose logs --tail=100 trading-api

# Health endpoints
curl http://localhost:8000/api/v1/health
curl http://localhost:8001/api/health
curl http://localhost:8006/api/health
curl http://localhost:8090/health
```

### Database Connectivity

```bash
# Connect to PostgreSQL
docker exec -it usdcop-postgres-timescale psql -U ${POSTGRES_USER} -d ${POSTGRES_DB}

# Check tables
\dt

# Check connections
SELECT count(*) FROM pg_stat_activity;
```

### Redis Connectivity

```bash
# Connect to Redis
docker exec -it usdcop-redis redis-cli -a ${REDIS_PASSWORD}

# Check memory
INFO memory

# Check connected clients
CLIENT LIST
```

---

## Common Issues

### 1. Service Won't Start

**Symptoms:**
- Container exits immediately
- `docker-compose up` hangs

**Solutions:**

```bash
# Check for port conflicts
netstat -tulpn | grep -E "8000|8001|8006|5432|6379"

# Free up ports if needed
sudo kill $(sudo lsof -t -i:8000)

# Check container logs
docker-compose logs trading-api

# Rebuild with no cache
docker-compose build --no-cache trading-api
docker-compose up trading-api
```

### 2. Database Connection Errors

**Symptoms:**
- `psycopg2.OperationalError: could not connect to server`
- `Connection refused`

**Solutions:**

```bash
# Check if PostgreSQL is running
docker-compose ps postgres

# Restart PostgreSQL
docker-compose restart postgres

# Wait for healthy status
docker-compose exec postgres pg_isready -U ${POSTGRES_USER}

# Check environment variables
cat .env | grep POSTGRES
```

### 3. Model Loading Errors

**Symptoms:**
- `ModelNotFoundError`
- `FileNotFoundError: models/onnx/`

**Solutions:**

```bash
# Check model files exist
ls -la models/onnx/

# Pull models from DVC
dvc pull models/

# Verify ONNX file
python -c "import onnxruntime; onnxruntime.InferenceSession('models/onnx/ppo_v2.onnx')"

# Check MLflow registry
curl http://localhost:5001/api/2.0/mlflow/registered-models/list
```

### 4. Feature Data Stale

**Symptoms:**
- `FeatureStaleError`
- Readiness score < 50%

**Solutions:**

```bash
# Check data freshness
curl http://localhost:8000/api/v1/readiness

# Trigger data pipeline
airflow dags trigger l0_macro_unified
airflow dags trigger l1_feature_refresh

# Check Feast materialization
feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)

# Force feature refresh
python scripts/refresh_features.py --force
```

### 5. High Latency / Slow Predictions

**Symptoms:**
- P99 latency > 200ms
- Timeout errors

**Solutions:**

```bash
# Check container resources
docker stats

# Increase resource limits (docker-compose.yml)
deploy:
  resources:
    limits:
      memory: 8G
      cpus: '4.0'

# Enable Redis caching
export REDIS_CACHE_ENABLED=true

# Check for memory leaks
docker exec -it usdcop-trading-api top -b -n 1
```

### 6. Rate Limiting Errors

**Symptoms:**
- `429 Too Many Requests`
- `RateLimitError`

**Solutions:**

```bash
# Check current rate limits
curl http://localhost:8000/api/v1/config/rate-limits

# Wait for rate limit reset (usually 60 seconds)

# Use API key for higher limits
curl -H "X-API-Key: your-key" http://localhost:8000/api/v1/predict

# Adjust rate limits (main.py)
rate_limit_requests_per_minute=200
```

### 7. Authentication Errors

**Symptoms:**
- `401 Unauthorized`
- `Invalid API Key`

**Solutions:**

```bash
# Generate new API key
python scripts/generate_api_key.py

# Check API key in database
docker exec -it usdcop-postgres-timescale psql -U ${POSTGRES_USER} -d ${POSTGRES_DB} \
  -c "SELECT key_prefix, is_active FROM api_keys;"

# Verify JWT secret is set
echo $JWT_SECRET

# Disable auth for testing (NOT for production)
export ENABLE_AUTH=false
```

### 8. Kill Switch Activated

**Symptoms:**
- Trading disabled
- `KillSwitchActiveError`

**Solutions:**

```bash
# Check kill switch status
curl http://localhost:8000/api/v1/operations/kill-switch

# Deactivate kill switch (with authorization)
curl -X POST http://localhost:8000/api/v1/operations/kill-switch/deactivate \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"reason": "manual reset after investigation"}'

# Check audit log
docker exec -it usdcop-postgres-timescale psql -U ${POSTGRES_USER} -d ${POSTGRES_DB} \
  -c "SELECT * FROM kill_switch_audit ORDER BY created_at DESC LIMIT 10;"
```

### 9. Memory Issues

**Symptoms:**
- `OutOfMemoryError`
- Container killed by OOM

**Solutions:**

```bash
# Check memory usage
docker stats --no-stream

# Increase container memory limits
# In docker-compose.yml:
deploy:
  resources:
    limits:
      memory: 8G

# Clear Redis cache
docker exec -it usdcop-redis redis-cli -a ${REDIS_PASSWORD} FLUSHDB

# Restart with fresh state
docker-compose down && docker-compose up -d
```

### 10. Airflow DAG Failures

**Symptoms:**
- DAG marked as failed
- Tasks in retry state

**Solutions:**

```bash
# Check Airflow UI
open http://localhost:8080

# View DAG logs
docker-compose logs airflow-scheduler | tail -100

# Clear failed task
docker exec -it usdcop-airflow-scheduler airflow tasks clear l0_macro_unified

# Manually trigger DAG
docker exec -it usdcop-airflow-scheduler airflow dags trigger l0_macro_unified

# Check task status
docker exec -it usdcop-airflow-scheduler airflow tasks state l0_macro_unified task_id 2026-01-17
```

---

## Log Analysis

### Finding Error Patterns

```bash
# Search for errors in all services
docker-compose logs --tail=500 | grep -i error

# Filter by specific error
docker-compose logs trading-api | grep "ConnectionError"

# Export logs for analysis
docker-compose logs --no-color > /tmp/all-logs.txt
```

### Important Log Locations

| Service | Log Location |
|---------|--------------|
| Trading API | `docker-compose logs trading-api` |
| PostgreSQL | `docker-compose logs postgres` |
| Airflow | `./airflow_logs/` |
| MLflow | `docker-compose logs mlflow` |
| Dashboard | `docker-compose logs dashboard` |

---

## Recovery Procedures

### Full System Restart

```bash
# Stop all services
docker-compose down

# Clear volumes (WARNING: deletes data)
docker-compose down -v

# Rebuild everything
docker-compose build --no-cache

# Start with fresh state
docker-compose up -d

# Wait for health checks
sleep 60
docker-compose ps
```

### Database Recovery

```bash
# Restore from backup
pg_restore -U ${POSTGRES_USER} -d ${POSTGRES_DB} /path/to/backup.dump

# Apply migrations
docker exec -it usdcop-postgres-timescale psql -U ${POSTGRES_USER} -d ${POSTGRES_DB} \
  -f /docker-entrypoint-initdb.d/01-essential-usdcop-init.sql
```

### Model Rollback

```bash
# List model versions
mlflow models list --name ppo_usdcop

# Promote previous version
mlflow models transition-stage --model ppo_usdcop --version 2 --stage Production

# Restart inference service
docker-compose restart mlops-inference-api
```

---

## Performance Tuning

### PostgreSQL

```sql
-- Check slow queries
SELECT query, calls, mean_time, total_time
FROM pg_stat_statements
ORDER BY mean_time DESC LIMIT 10;

-- Add indexes
CREATE INDEX CONCURRENTLY idx_features_timestamp ON features(timestamp);

-- Vacuum analyze
VACUUM ANALYZE features;
```

### Redis

```bash
# Check memory usage
redis-cli -a ${REDIS_PASSWORD} INFO memory

# Configure maxmemory policy
redis-cli -a ${REDIS_PASSWORD} CONFIG SET maxmemory-policy allkeys-lru
```

---

## Getting Help

1. **Check Documentation:**
   - `docs/ARCHITECTURE.md` - System architecture
   - `docs/REPRODUCIBILITY.md` - Environment setup
   - `docs/INCIDENT_RESPONSE_PLAYBOOK.md` - Incident handling

2. **Search Issues:**
   - GitHub Issues: https://github.com/your-org/usdcop-rl-models/issues

3. **Collect Diagnostics:**
   ```bash
   # Generate diagnostic report
   python scripts/collect_diagnostics.py > diagnostics.txt
   ```

4. **Contact Support:**
   - Email: trading-support@example.com
   - Slack: #trading-support

---

*Document Version: 1.0.0*
*Last Updated: 2026-01-17*
