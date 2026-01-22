# Feast Troubleshooting Runbook
## USD/COP RL Trading System

**Contract**: FEAST-30
**Version**: 1.0.0
**Last Updated**: 2026-01-17

---

## Quick Reference

| Issue | Severity | Section |
|-------|----------|---------|
| Redis connection failed | P0 | [Redis Issues](#redis-connection-issues) |
| Features not materializing | P1 | [Materialization Issues](#materialization-issues) |
| Stale features | P1 | [Stale Feature Issues](#stale-feature-issues) |
| Schema mismatch | P2 | [Schema Issues](#schema-mismatch-issues) |
| Registry corruption | P2 | [Registry Issues](#registry-issues) |

---

## Redis Connection Issues

### Symptoms
- `ConnectionError: Error connecting to Redis`
- `ConnectionRefusedError: [Errno 111] Connection refused`
- Feature retrieval returns None or empty

### Diagnostic Steps

```bash
# 1. Check Redis container status
docker-compose ps redis

# Expected: State = Up (healthy)

# 2. Test Redis connectivity
docker exec -it usdcop-redis redis-cli ping
# Expected: PONG

# 3. Test with password
docker exec -it usdcop-redis redis-cli -a ${REDIS_PASSWORD} ping
# Expected: PONG

# 4. Check Redis logs
docker-compose logs --tail=50 redis

# 5. Verify Redis port is accessible
nc -zv localhost 6379
```

### Resolution

**If Redis is down:**
```bash
docker-compose restart redis
# Wait for healthy status
docker-compose ps redis
```

**If password authentication fails:**
```bash
# Check password in .env
cat .env | grep REDIS_PASSWORD

# Verify in container
docker exec -it usdcop-redis cat /run/secrets/redis_password 2>/dev/null || \
    docker exec -it usdcop-redis printenv REDIS_PASSWORD
```

**If port is blocked:**
```bash
# Check for port conflicts
lsof -i :6379

# Restart with clean network
docker-compose down
docker-compose up -d redis
```

---

## Materialization Issues

### Symptoms
- `FeatureNotFoundError` during inference
- Materialization DAG fails
- Empty feature values returned

### Diagnostic Steps

```bash
# 1. Check materialization DAG status
airflow dags list-runs -d v3.l1b_feast_materialize --limit 5

# 2. Check Feast registry
cd feature_repo
feast registry-dump

# 3. Verify feature views exist
feast feature-views list

# 4. Check last materialization time
feast materialize-incremental --help
# Look for materialization history in registry
```

### Resolution

**Manual materialization:**
```bash
cd feature_repo

# Apply feature definitions first
feast apply

# Run incremental materialization
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

**Full re-materialization:**
```bash
cd feature_repo

# Clear and re-materialize
feast materialize \
    --start-time $(date -u -d "7 days ago" +"%Y-%m-%dT%H:%M:%S") \
    --end-time $(date -u +"%Y-%m-%dT%H:%M:%S")
```

**DAG troubleshooting:**
```bash
# Check DAG logs
airflow tasks logs v3.l1b_feast_materialize materialize_features -1

# Clear failed task
airflow tasks clear v3.l1b_feast_materialize -t materialize_features -y
```

---

## Stale Feature Issues

### Symptoms
- Features have old timestamps
- Inference using outdated data
- TTL expired warnings

### Diagnostic Steps

```bash
# 1. Check feature freshness in Redis
docker exec -it usdcop-redis redis-cli --scan --pattern "feast:*" | head -5

# 2. Get TTL for a feature key
docker exec -it usdcop-redis redis-cli TTL "feast:technical_features:USDCOP"

# 3. Check when features were last materialized
cd feature_repo
feast registry-dump | grep -A5 "last_updated"

# 4. Verify upstream L1 pipeline ran
airflow dags list-runs -d v3.l1_feature_refresh --limit 3
```

### Resolution

**Force fresh materialization:**
```bash
cd feature_repo

# Re-materialize recent data
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

**Increase TTL if needed:**
```python
# In feature_repo/features.py
technical_features = FeatureView(
    name="technical_features",
    ttl=timedelta(hours=48),  # Increase from 24 to 48
    ...
)
```

**Trigger upstream pipeline:**
```bash
airflow dags trigger v3.l1_feature_refresh
# Wait for completion
airflow dags trigger v3.l1b_feast_materialize
```

---

## Schema Mismatch Issues

### Symptoms
- `ValueError: Feature count mismatch: expected 15, got X`
- `KeyError: 'feature_name'`
- Training/inference parity broken

### Diagnostic Steps

```bash
# 1. Compare feature definitions with contract
python -c "
from src.core.contracts.feature_contract import FEATURE_ORDER
print('Contract features:', len(FEATURE_ORDER))
print(FEATURE_ORDER)
"

# 2. Check Feast feature views
cd feature_repo
feast feature-views list
feast feature-services describe observation_15d

# 3. Validate schema
feast validate
```

### Resolution

**Update feature definitions:**
```bash
cd feature_repo

# Edit features.py to match contract
# Then apply changes
feast apply --force

# Re-materialize
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

**Verify contract alignment:**
```python
# Run validation script
python scripts/validate_feature_contract.py
```

---

## Registry Issues

### Symptoms
- `RegistryNotFoundError`
- Corrupt registry database
- `sqlite3.DatabaseError`

### Diagnostic Steps

```bash
# 1. Check registry file
ls -la feature_repo/data/registry.db

# 2. Validate registry
cd feature_repo
sqlite3 data/registry.db ".tables"

# 3. Backup current registry
cp data/registry.db data/registry.db.backup
```

### Resolution

**Rebuild registry:**
```bash
cd feature_repo

# Remove corrupt registry
mv data/registry.db data/registry.db.corrupt

# Re-apply all definitions
feast apply

# Re-materialize all data
feast materialize \
    --start-time $(date -u -d "7 days ago" +"%Y-%m-%dT%H:%M:%S") \
    --end-time $(date -u +"%Y-%m-%dT%H:%M:%S")
```

**Restore from backup:**
```bash
cd feature_repo
cp data/registry.db.backup data/registry.db
feast validate
```

---

## Fallback Procedures

### When Feast is Completely Unavailable

The system includes automatic fallback to `CanonicalFeatureBuilder`:

```python
# In inference code
try:
    features = feast_service.get_online_features(entity)
except Exception as e:
    logger.warning(f"Feast unavailable: {e}")
    # Automatic fallback to direct calculation
    features = canonical_builder.build_observation(timestamp)
```

### Manual Fallback Activation

```bash
# Set environment variable to force fallback
export FEAST_FALLBACK_ENABLED=true

# Restart inference services
docker-compose restart mlops-inference-api backtest-api
```

---

## Monitoring Checklist

### Daily Checks

- [ ] Redis container healthy
- [ ] Last materialization < 1 hour ago
- [ ] No stale feature warnings in logs
- [ ] Inference latency < 50ms

### Weekly Checks

- [ ] Registry size reasonable (< 100MB)
- [ ] Redis memory usage < 80%
- [ ] Feast version compatibility
- [ ] Feature drift metrics

### Commands for Monitoring

```bash
# Check all Feast-related containers
docker-compose ps | grep -E "redis|feast"

# Monitor Redis memory
docker exec -it usdcop-redis redis-cli INFO memory | grep used_memory_human

# Check materialization health
airflow dags list-runs -d v3.l1b_feast_materialize --state failed --limit 5
```

---

## Escalation

| Level | Condition | Action |
|-------|-----------|--------|
| L1 | Transient errors, auto-recovered | Monitor |
| L2 | Materialization failed 3+ times | Manual intervention |
| L3 | Redis down > 5 minutes | Restart infrastructure |
| P0 | Fallback also failing | Emergency maintenance |

### Contact

- **On-call**: Check PagerDuty/Slack
- **Documentation**: This runbook
- **Source code**: `src/feature_store/`

---

*Document maintained by USDCOP Trading Team*
