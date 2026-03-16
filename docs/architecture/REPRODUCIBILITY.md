# Reproducibility Guide
## USD/COP RL Trading System

**Version:** 1.0.0
**Date:** 2026-01-17

---

## Overview

This document provides comprehensive instructions for reproducing the USD/COP RL Trading System from source. Full reproducibility ensures that anyone can:

1. Recreate the exact development environment
2. Reproduce model training with identical results
3. Reconstruct datasets from raw data
4. Deploy identical production systems

---

## 1. Environment Setup

### 1.1 Python Version

The system requires Python 3.11.x. The exact version is pinned in CI and should be used for development.

```bash
# Check Python version
python --version
# Expected: Python 3.11.x

# Using pyenv (recommended)
pyenv install 3.11.7
pyenv local 3.11.7
```

### 1.2 Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Verify
which python
# Should point to .venv/bin/python
```

### 1.3 Dependencies

Install pinned dependencies from the lock file:

```bash
# Install from lock file (exact versions)
pip install -r requirements.lock

# Or install with optional groups
pip install -e ".[all]"
```

### 1.4 Verify Installation

```bash
# Run reproducibility check
python -c "
import sys
import numpy as np
import torch
import stable_baselines3

print(f'Python: {sys.version}')
print(f'NumPy: {np.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'SB3: {stable_baselines3.__version__}')

# Verify deterministic operations
np.random.seed(42)
print(f'NumPy seed test: {np.random.rand()}')

torch.manual_seed(42)
print(f'PyTorch seed test: {torch.rand(1).item()}')
"
```

---

## 2. Data Reproducibility

### 2.1 DVC Setup

All datasets are version-controlled with DVC:

```bash
# Initialize DVC (already done)
dvc init

# Configure remote (MinIO)
dvc remote add -d minio s3://usdcop-dvc
dvc remote modify minio endpointurl http://localhost:9000

# Pull data for a specific version
dvc checkout

# Pull all data
dvc pull
```

### 2.2 Data Version Tags

| Tag | Description | Date |
|-----|-------------|------|
| `data-v1.0` | Initial dataset | 2025-06-01 |
| `data-v1.1` | Added macro features | 2025-09-15 |
| `data-v2.0` | Production dataset | 2026-01-01 |

```bash
# Checkout specific data version
git checkout data-v2.0
dvc checkout
```

### 2.3 Data Reconstruction

To reconstruct datasets from raw sources:

```bash
# Step 1: Run L0 pipeline (raw data ingestion)
airflow dags trigger l0_macro_unified

# Step 2: Run L1 pipeline (feature engineering)
airflow dags trigger l1_feature_refresh

# Step 3: Build training dataset
python scripts/prepare_training_data.py \
    --start-date 2018-01-01 \
    --end-date 2023-12-31 \
    --output data/processed/training_data.parquet

# Step 4: Verify hash
python -c "
import hashlib
import pathlib
content = pathlib.Path('data/processed/training_data.parquet').read_bytes()
print(f'SHA256: {hashlib.sha256(content).hexdigest()}')
"
```

### 2.4 Data Checksums

Critical data files have checksums recorded in `dvc.lock`:

```yaml
# dvc.lock (excerpt)
stages:
  prepare_data:
    outs:
      - path: data/processed/training_data.parquet
        md5: abc123def456...
        size: 12345678
```

---

## 3. Model Training Reproducibility

### 3.1 Random Seeds

All random seeds are configured in a central location:

```python
# src/training/train_ssot.py

import random
import numpy as np
import torch

def set_all_seeds(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # PyTorch deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Environment variables for additional determinism
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Usage
set_all_seeds(42)
```

### 3.2 Training Command

```bash
# Full reproducible training
python scripts/train_with_mlflow.py \
    --seed 42 \
    --data data/processed/training_data.parquet \
    --config config/training/ppo_config.yaml \
    --experiment usdcop-production \
    --run-name ppo-v2-reproducible
```

### 3.3 Hyperparameters

All hyperparameters are stored in `params.yaml`:

```yaml
# params.yaml
training:
  seed: 42
  algorithm: PPO
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
  total_timesteps: 1000000
```

### 3.4 MLflow Tracking

Every training run is logged to MLflow:

```bash
# View training history
mlflow ui --port 5000

# Reproduce a specific run
mlflow run . --experiment-name usdcop-production \
    --run-id abc123def456
```

---

## 4. Model Artifacts

### 4.1 Model Registry

Models are versioned in the MLflow Model Registry:

```python
import mlflow

# Load production model
model = mlflow.pyfunc.load_model("models:/ppo_usdcop/Production")

# Load specific version
model = mlflow.pyfunc.load_model("models:/ppo_usdcop/3")
```

### 4.2 ONNX Export

For inference reproducibility, models are exported to ONNX:

```bash
# Export to ONNX
python scripts/export_onnx.py \
    --model-uri models:/ppo_usdcop/Production \
    --output models/onnx/ppo_v2.onnx

# Verify ONNX model
python -c "
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('models/onnx/ppo_v2.onnx')
input_name = session.get_inputs()[0].name

# Test inference
test_input = np.random.randn(1, 34).astype(np.float32)
result = session.run(None, {input_name: test_input})
print(f'Output shape: {result[0].shape}')
"
```

### 4.3 Model Hashes

Model files include SHA256 checksums:

```bash
# Generate model hash
sha256sum models/onnx/ppo_v2.onnx > models/onnx/ppo_v2.onnx.sha256

# Verify model integrity
sha256sum -c models/onnx/ppo_v2.onnx.sha256
```

---

## 5. Infrastructure Reproducibility

### 5.1 Docker Images

All services use pinned Docker images:

```yaml
# docker-compose.yml (excerpt)
services:
  postgres:
    image: timescale/timescaledb:2.13.0-pg15
  redis:
    image: redis:7.2-alpine
  minio:
    image: minio/minio:RELEASE.2024-01-05T22-17-24Z
```

### 5.2 Build Reproducibility

```bash
# Build with specific cache settings
docker build \
    --no-cache \
    --build-arg BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
    --build-arg GIT_SHA=$(git rev-parse HEAD) \
    -t usdcop/inference-api:$(git rev-parse --short HEAD) \
    -f services/inference_api/Dockerfile .
```

### 5.3 Environment Variables

Required environment variables are documented in `.env.example`:

```bash
# Copy and configure
cp .env.example .env

# Verify all variables are set
python scripts/validate_env.py
```

---

## 6. Testing Reproducibility

### 6.1 Deterministic Tests

Tests use fixed seeds for reproducibility:

```python
# conftest.py
import pytest

@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed before each test."""
    import random
    import numpy as np
    import torch

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    yield
```

### 6.2 Test Commands

```bash
# Run all tests with reproducibility
pytest tests/ -v --tb=short

# Run with specific seed (override)
RANDOM_SEED=12345 pytest tests/

# Generate coverage report
pytest tests/ --cov=src --cov-report=html
```

---

## 7. Verification Checklist

Use this checklist to verify full reproducibility:

### Environment
- [ ] Python 3.11.x installed
- [ ] Virtual environment created
- [ ] Dependencies installed from lock file
- [ ] Environment variables configured

### Data
- [ ] DVC remote configured
- [ ] Data pulled with `dvc pull`
- [ ] Data checksums verified
- [ ] Feature timestamps validated

### Model
- [ ] Seeds set correctly (42)
- [ ] Hyperparameters from params.yaml
- [ ] Training logged to MLflow
- [ ] Model exported to ONNX
- [ ] Model hash verified

### Infrastructure
- [ ] Docker images pinned
- [ ] Secrets configured (not in Git)
- [ ] Database migrations applied
- [ ] Health checks passing

### Tests
- [ ] All unit tests passing
- [ ] All integration tests passing
- [ ] Coverage threshold met (70%)

---

## 8. Troubleshooting

### Non-Deterministic Behavior

If you observe non-deterministic results:

1. **Check CUDA operations**: Set `CUBLAS_WORKSPACE_CONFIG=:4096:8`
2. **Verify seeds**: Print seeds at start of training
3. **Check data order**: Ensure consistent file reading order
4. **GPU vs CPU**: Results may differ; stick to one

### Data Version Mismatch

```bash
# Check current data version
dvc status

# Force checkout to correct version
dvc checkout --force

# Verify file hashes
dvc diff
```

### Model Loading Errors

```bash
# Check model file integrity
sha256sum -c models/onnx/ppo_v2.onnx.sha256

# Verify ONNX compatibility
python -c "import onnx; onnx.checker.check_model('models/onnx/ppo_v2.onnx')"
```

---

## 9. References

- [DVC Documentation](https://dvc.org/doc)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [PyTorch Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)
- [Stable-Baselines3 Reproducibility](https://stable-baselines3.readthedocs.io/en/master/guide/reproducibility.html)

---

*Document Version: 1.0.0*
*Last Updated: 2026-01-17*
*Author: Trading Team*
