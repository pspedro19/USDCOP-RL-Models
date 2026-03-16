# MLflow Integration Guide
## USD/COP RL Trading System

**Contract**: INTDOC-09
**Version**: 1.0.0
**Last Updated**: 2026-01-17

---

## Overview

MLflow is used in the USDCOP Trading System for:

- **Experiment tracking**: Log parameters, metrics, and artifacts
- **Model registry**: Version and stage models (Staging → Production)
- **Artifact storage**: Store models in MinIO
- **Reproducibility**: Track all training metadata

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MLFLOW ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐         ┌─────────────────┐               │
│  │  MLflow Server  │────────▶│   PostgreSQL    │               │
│  │    :5001        │         │   (Backend)     │               │
│  └────────┬────────┘         └─────────────────┘               │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │     MinIO       │                                           │
│  │   (Artifacts)   │                                           │
│  │  s3://mlflow/   │                                           │
│  └─────────────────┘                                           │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   MODEL REGISTRY                         │   │
│  │  ┌──────┐   ┌─────────┐   ┌────────────┐   ┌──────────┐│   │
│  │  │ None │──▶│ Staging │──▶│ Production │──▶│ Archived ││   │
│  │  └──────┘   └─────────┘   └────────────┘   └──────────┘│   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration

### Docker Compose

```yaml
mlflow:
  image: ghcr.io/mlflow/mlflow:v2.10.2
  container_name: trading-mlflow
  depends_on:
    minio:
      condition: service_healthy
  environment:
    - MLFLOW_S3_ENDPOINT_URL=http://minio:9000
    - AWS_ACCESS_KEY_ID=${MINIO_ACCESS_KEY}
    - AWS_SECRET_ACCESS_KEY=${MINIO_SECRET_KEY}
  command: |
    mlflow server
    --backend-store-uri postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
    --default-artifact-root s3://mlflow/
    --host 0.0.0.0
    --port 5000
  ports:
    - "5001:5000"
```

### Environment Variables

```bash
# MLflow tracking
export MLFLOW_TRACKING_URI=http://localhost:5001

# MinIO for artifacts
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=your_minio_key
export AWS_SECRET_ACCESS_KEY=your_minio_secret
```

---

## Training Integration

### Basic Usage

```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5001")

# Set experiment
mlflow.set_experiment("ppo_usdcop")

# Start run
with mlflow.start_run(run_name="training_v1"):
    # Log parameters
    mlflow.log_params({
        "total_timesteps": 500000,
        "learning_rate": 3e-4,
        "batch_size": 64,
    })

    # Train model
    model = train_model(...)

    # Log metrics
    mlflow.log_metrics({
        "best_reward": 150.0,
        "training_duration": 3600,
    })

    # Log model artifact
    mlflow.log_artifact("models/final_model.zip")
```

### Hash Logging (CTR-HASH-001)

```python
import hashlib

# Log all reproducibility hashes
mlflow.log_params({
    "dataset_hash": compute_file_hash(dataset_path)[:16],
    "norm_stats_hash": compute_json_hash(norm_stats_path)[:16],
    "git_commit_sha": get_git_commit()[:12],
})

mlflow.set_tags({
    "dataset_hash_full": compute_file_hash(dataset_path),
    "norm_stats_hash_full": compute_json_hash(norm_stats_path),
    "git_commit_sha": get_git_commit(),
    "dvc_version": get_dvc_version(),
    "reproducibility_verified": "true",
})
```

### Model Signature

```python
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
import numpy as np

# Define input/output schema
input_schema = Schema([
    TensorSpec(np.dtype("float32"), (-1, 15), "observation")
])
output_schema = Schema([
    TensorSpec(np.dtype("float32"), (-1, 1), "action")
])

signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Log signature
mlflow.log_dict({
    "observation_dim": 15,
    "action_dim": 1,
    "feature_order": list(FEATURE_ORDER),
}, "model_signature.json")
```

---

## Model Registry

### Stages

| Stage | Purpose | Access |
|-------|---------|--------|
| None | Just registered | Development |
| Staging | Testing/validation | QA |
| Production | Live trading | Inference |
| Archived | Deprecated | Historical |

### Promotion Workflow

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register model
result = client.create_registered_model("ppo_usdcop")

# Create version from run
model_version = client.create_model_version(
    name="ppo_usdcop",
    source=f"runs:/{run_id}/model",
    run_id=run_id
)

# Transition to Staging
client.transition_model_version_stage(
    name="ppo_usdcop",
    version=model_version.version,
    stage="Staging"
)

# After validation, promote to Production
client.transition_model_version_stage(
    name="ppo_usdcop",
    version=model_version.version,
    stage="Production"
)
```

### Promotion Validations

The `scripts/promote_model.py` requires:

1. **Smoke test passed**: Model can load and predict
2. **Dataset hash match**: Training data hash matches expected
3. **Staging time**: Model spent minimum 24 hours in Staging

---

## Loading Models

### For Inference

```python
import mlflow

# Load production model
model = mlflow.pyfunc.load_model("models:/ppo_usdcop/Production")

# Or load specific version
model = mlflow.pyfunc.load_model("models:/ppo_usdcop/3")

# Or load from run
model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
```

### For Stable-Baselines3

```python
from stable_baselines3 import PPO

# Download artifact and load
client = MlflowClient()
local_path = client.download_artifacts(run_id, "model/final_model.zip")
model = PPO.load(local_path)
```

---

## Querying Experiments

### Search Runs

```python
import mlflow

# Find best runs
runs = mlflow.search_runs(
    experiment_names=["ppo_usdcop"],
    filter_string="metrics.best_reward > 100",
    order_by=["metrics.best_reward DESC"],
    max_results=10
)

# Get run details
for _, run in runs.iterrows():
    print(f"Run: {run.run_id}, Reward: {run['metrics.best_reward']}")
```

### Compare Runs

```python
# Get parameters across runs
params_df = runs[['params.learning_rate', 'params.batch_size', 'metrics.best_reward']]
print(params_df)
```

---

## Troubleshooting

### Common Issues

#### 1. Connection Refused

```bash
# Check MLflow is running
docker-compose ps mlflow

# Check health
curl http://localhost:5001/health

# View logs
docker-compose logs mlflow
```

#### 2. Artifact Upload Failed

```bash
# Check MinIO connectivity
mc alias set minio http://localhost:9000 ${MINIO_ACCESS_KEY} ${MINIO_SECRET_KEY}
mc ls minio/mlflow/

# Check MLflow S3 settings
docker exec trading-mlflow printenv | grep AWS
```

#### 3. Model Not Found

```python
# List registered models
client = MlflowClient()
for rm in client.search_registered_models():
    print(rm.name)

# List versions
for mv in client.search_model_versions(f"name='ppo_usdcop'"):
    print(f"Version {mv.version}: {mv.current_stage}")
```

---

## Best Practices

1. **Always log hashes** for all artifacts (CTR-HASH-001)
2. **Use model signatures** for input validation
3. **Follow staging workflow** - never promote directly to Production
4. **Log environment info** (Python version, package versions)
5. **Use meaningful run names** with timestamps
6. **Archive old models** rather than deleting

---

## Related Documentation

- [MLflow Official Docs](https://mlflow.org/docs/latest/)
- [Model Promotion Script](../scripts/promote_model.py)
- [Hash Reconciliation](../scripts/validate_hash_reconciliation.py)
- [DVC Integration Guide](./DVC_INTEGRATION_GUIDE.md)

---

*Document maintained by USDCOP Trading Team*
