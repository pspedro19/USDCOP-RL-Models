# DVC Integration Guide
## USD/COP RL Trading System

**Contract**: INTDOC-10, DVC-40
**Version**: 1.0.0
**Last Updated**: 2026-01-17

---

## Overview

DVC (Data Version Control) is used in the USDCOP Trading System for:

- **Dataset versioning**: Track training data versions
- **Pipeline reproducibility**: Ensure consistent training
- **Remote storage**: Store large files in MinIO/S3
- **Experiment tracking**: Link data to model versions

---

## Configuration

### DVC Remote Configuration

**File**: `.dvc/config`

```ini
[core]
    remote = minio
    autostage = true

[remote "minio"]
    url = s3://dvc-storage
    endpointurl = http://minio:9000

[remote "s3_backup"]
    url = s3://usdcop-dvc-backup
    region = us-east-1
```

### Environment Variables

```bash
# MinIO credentials (used by DVC)
export AWS_ACCESS_KEY_ID=your_minio_access_key
export AWS_SECRET_ACCESS_KEY=your_minio_secret_key

# For S3 backup
export AWS_DEFAULT_REGION=us-east-1
```

---

## Pipeline Stages

### DVC Pipeline (`dvc.yaml`)

```yaml
stages:
  prepare_data:
    cmd: python scripts/prepare_training_data.py
    deps:
      - scripts/prepare_training_data.py
      - data/raw/
    outs:
      - data/pipeline/07_output/datasets_5min/RL_DS3_MACRO_CORE.csv

  compute_norm_stats:
    cmd: python scripts/compute_norm_stats.py
    deps:
      - data/pipeline/07_output/datasets_5min/RL_DS3_MACRO_CORE.csv
    outs:
      - config/norm_stats.json

  train_model:
    cmd: python scripts/train_with_mlflow.py
    deps:
      - data/pipeline/07_output/datasets_5min/RL_DS3_MACRO_CORE.csv
      - config/norm_stats.json
      - src/training/
    params:
      - params.yaml:
    outs:
      - models/ppo_production/final_model.zip
    metrics:
      - metrics/training_metrics.json:
          cache: false

  validate_model:
    cmd: python scripts/validate_model.py
    deps:
      - models/ppo_production/final_model.zip
      - config/norm_stats.json
    metrics:
      - metrics/validation_metrics.json:
          cache: false

  promote_model:
    cmd: python scripts/promote_model.py
    deps:
      - models/ppo_production/final_model.zip
      - metrics/validation_metrics.json
```

---

## Common Commands

### Daily Operations

```bash
# Check status
dvc status

# Pull latest data
dvc pull

# Push local changes
dvc push

# Reproduce pipeline
dvc repro
```

### Dataset Management

```bash
# Track a new dataset
dvc add data/new_dataset.csv

# Commit DVC files
git add data/new_dataset.csv.dvc .gitignore
git commit -m "Add new dataset"

# Push data to remote
dvc push
```

### Pipeline Operations

```bash
# Run full pipeline
dvc repro

# Run specific stage
dvc repro train_model

# Force re-run
dvc repro --force

# Show DAG
dvc dag
```

### Version Control

```bash
# List tracked files
dvc list . --dvc-only

# Show file history
dvc diff HEAD~1

# Checkout specific version
git checkout <commit> -- dvc.lock
dvc checkout
```

---

## Integration with Training

### Before Training (L3 DAG)

The L3 training DAG now includes DVC checkout:

```python
# In l3_model_training.py
def dvc_checkout_dataset(**context):
    """Checkout dataset from DVC before training."""
    subprocess.run(['dvc', 'pull', '-r', 'minio'], check=True)
    subprocess.run(['dvc', 'checkout'], check=True)
```

### Logging to MLflow

```python
# Log DVC hash to MLflow
dvc_hash = get_dvc_dataset_hash(dataset_path)
mlflow.set_tag("dvc_dataset_hash", dvc_hash)

# Log git commit
git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
mlflow.set_tag("git_commit_sha", git_commit)
```

---

## Publishing Datasets

### Automated Publishing

```bash
# Use the publish script
./scripts/publish_dataset.sh -t v1.0.0 -m "Release training dataset"
```

### Manual Publishing

```bash
# 1. Run pipeline to ensure consistency
dvc repro

# 2. Add and commit DVC files
git add dvc.lock *.dvc
git commit -m "Update dataset version"

# 3. Tag release
git tag -a v1.0.0 -m "Dataset release v1.0.0"

# 4. Push to remotes
git push origin main --tags
dvc push -r minio
```

---

## Rollback Procedures

### Rollback to Previous Version

```bash
# Use the rollback script
./scripts/rollback_dataset.sh -t v0.9.0

# Or manually
git checkout v0.9.0 -- dvc.lock *.dvc
dvc checkout
dvc pull
```

### View Available Versions

```bash
# List tags
git tag -l

# Show DVC-related commits
git log --oneline -- "*.dvc" "dvc.lock"
```

---

## Troubleshooting

### Common Issues

#### 1. Remote Not Accessible

```bash
# Check remote configuration
dvc remote list

# Test MinIO connectivity
mc alias set minio http://localhost:9000 ${MINIO_ACCESS_KEY} ${MINIO_SECRET_KEY}
mc ls minio/dvc-storage/

# Try backup remote
dvc pull -r s3_backup
```

#### 2. Cache Corruption

```bash
# Clear local cache
rm -rf .dvc/cache

# Re-pull data
dvc pull
```

#### 3. Lock File Conflicts

```bash
# Reset lock file
git checkout origin/main -- dvc.lock
dvc checkout
```

#### 4. Pipeline Not Reproducing

```bash
# Check what changed
dvc status

# Force re-run
dvc repro --force
```

---

## Best Practices

1. **Always run `dvc repro` before training** to ensure data consistency
2. **Tag releases** with semantic versioning
3. **Push to both remotes** (minio and s3_backup) for redundancy
4. **Log DVC hashes to MLflow** for traceability
5. **Use `dvc.lock`** as the source of truth for reproducibility
6. **Never modify tracked data manually** - always use pipeline

---

## Related Documentation

- [DVC Official Docs](https://dvc.org/doc)
- [Dataset Publishing Script](../scripts/publish_dataset.sh)
- [Dataset Rollback Script](../scripts/rollback_dataset.sh)
- [MLflow Integration Guide](./MLFLOW_INTEGRATION_GUIDE.md)

---

*Document maintained by USDCOP Trading Team*
