# Sync Recovery Runbook
## USD/COP RL Trading System

**Contract**: SYNC-19
**Version**: 1.0.0
**Last Updated**: 2026-01-17

---

## Overview

This runbook provides procedures for recovering from synchronization failures between system components:

- PostgreSQL ↔ Feast (Materialization)
- DVC ↔ MinIO (Dataset versioning)
- MLflow ↔ MinIO (Model artifacts)
- Git ↔ DVC (Version control)

---

## Quick Reference

| Sync Type | Source of Truth | Recovery Script |
|-----------|-----------------|-----------------|
| Features | PostgreSQL | `feast materialize-incremental` |
| Datasets | DVC/Git | `scripts/rollback_dataset.sh` |
| Models | MLflow Registry | See MLflow recovery |
| Artifacts | MinIO | `dvc push` / `dvc pull` |

---

## PostgreSQL → Feast Sync Recovery

### Symptoms
- Inference using stale features
- Materialization DAG failing
- Redis features don't match PostgreSQL

### Recovery Steps

```bash
# Step 1: Verify PostgreSQL has fresh data
psql -h localhost -U ${POSTGRES_USER} -d ${POSTGRES_DB} -c "
SELECT MAX(time) as latest_bar,
       NOW() - MAX(time) as age
FROM usdcop_m5_ohlcv;
"

# Step 2: Check Feast registry health
cd feature_repo
feast validate

# Step 3: Clear stale Redis data (optional)
docker exec -it usdcop-redis redis-cli FLUSHDB

# Step 4: Re-materialize features
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")

# Step 5: Verify sync
python -c "
from feast import FeatureStore
store = FeatureStore(repo_path='feature_repo')
features = store.get_online_features(
    features=['technical_features:rsi_9'],
    entity_rows=[{'symbol': 'USDCOP'}]
).to_dict()
print('Features:', features)
"
```

---

## DVC → MinIO Sync Recovery

### Symptoms
- `dvc pull` fails with "file not found"
- Dataset hashes don't match
- Training uses wrong data version

### Recovery Steps

```bash
# Step 1: Check DVC remote status
dvc remote list
dvc status -c -r minio

# Step 2: Verify MinIO accessibility
mc alias set minio http://localhost:9000 ${MINIO_ACCESS_KEY} ${MINIO_SECRET_KEY}
mc ls minio/dvc-storage/

# Step 3: If data missing from MinIO, push from local cache
dvc push -r minio

# Step 4: If local cache is empty, try backup remote
dvc pull -r s3_backup

# Step 5: Verify data integrity
dvc status
sha256sum data/pipeline/07_output/datasets_5min/RL_DS3_MACRO_CORE.csv
```

### Full Recovery from Git History

```bash
# Find last known good version
git log --oneline -- "*.dvc" "dvc.lock" | head -10

# Checkout that version
git checkout <commit_sha> -- dvc.lock *.dvc

# Pull data for that version
dvc checkout
dvc pull -r minio

# Verify
dvc status
```

---

## MLflow → MinIO Sync Recovery

### Symptoms
- Model artifacts missing
- "Artifact not found" errors
- Model loading fails

### Recovery Steps

```bash
# Step 1: Check MLflow tracking server
curl http://localhost:5001/health

# Step 2: List runs with artifacts
python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5001')
runs = mlflow.search_runs(experiment_ids=['0'])
print(runs[['run_id', 'artifact_uri']].head())
"

# Step 3: Check MinIO bucket
mc ls minio/mlflow/

# Step 4: If artifacts missing, re-log from local model
python scripts/reupload_mlflow_artifacts.py --run-id <run_id>

# Step 5: Verify artifact accessibility
python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5001')
artifacts = mlflow.artifacts.list_artifacts('<run_id>')
print('Artifacts:', [a.path for a in artifacts])
"
```

---

## Git ↔ DVC Sync Recovery

### Symptoms
- DVC files not tracked by Git
- `dvc.lock` out of sync
- Pipeline reproducibility broken

### Recovery Steps

```bash
# Step 1: Check Git status
git status

# Step 2: Check DVC status
dvc status

# Step 3: If dvc.lock is dirty, re-run pipeline
dvc repro

# Step 4: Stage DVC files
git add dvc.lock *.dvc

# Step 5: Commit changes
git commit -m "Sync DVC files with pipeline state"

# Step 6: Push both Git and DVC
git push origin main
dvc push -r minio
```

---

## Full System Sync Recovery

### When to Use
- After major infrastructure failure
- After disaster recovery
- After data corruption

### Steps

```bash
#!/bin/bash
# Full system sync recovery script

echo "=== FULL SYSTEM SYNC RECOVERY ==="

# 1. Ensure all services are running
echo "[1/7] Checking services..."
docker-compose ps

# 2. Verify PostgreSQL connectivity
echo "[2/7] Checking PostgreSQL..."
pg_isready -h localhost -p 5432

# 3. Verify MinIO connectivity
echo "[3/7] Checking MinIO..."
mc alias set minio http://localhost:9000 ${MINIO_ACCESS_KEY} ${MINIO_SECRET_KEY}
mc admin info minio

# 4. Verify Redis connectivity
echo "[4/7] Checking Redis..."
docker exec -it usdcop-redis redis-cli ping

# 5. Sync DVC data
echo "[5/7] Syncing DVC..."
dvc checkout
dvc pull -r minio || dvc pull -r s3_backup

# 6. Sync Feast features
echo "[6/7] Syncing Feast..."
cd feature_repo
feast apply
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
cd ..

# 7. Verify MLflow
echo "[7/7] Verifying MLflow..."
curl http://localhost:5001/health

echo "=== SYNC RECOVERY COMPLETE ==="
```

---

## Verification Checklist

After recovery, verify:

- [ ] PostgreSQL has latest OHLCV data (< 10 min old)
- [ ] Feast features are materialized (check Redis)
- [ ] DVC status is clean (`dvc status` returns empty)
- [ ] MLflow can list experiments and runs
- [ ] Inference API returns valid predictions
- [ ] All Docker healthchecks pass

### Verification Script

```bash
# Quick verification
python -c "
import requests
import json

# Check all services
services = {
    'postgres': 'pg_isready check passed',
    'redis': requests.get('http://localhost:6379').status_code if False else 'manual check',
    'mlflow': requests.get('http://localhost:5001/health').status_code,
    'inference': requests.get('http://localhost:8090/health').status_code,
}

for svc, status in services.items():
    print(f'{svc}: {status}')
"
```

---

## Escalation

| Severity | Condition | Action |
|----------|-----------|--------|
| Low | Single service out of sync | Follow runbook |
| Medium | Multiple services affected | Notify team |
| High | Full system sync failure | Emergency meeting |
| Critical | Data loss detected | Restore from backup |

---

*Document maintained by USDCOP Trading Team*
