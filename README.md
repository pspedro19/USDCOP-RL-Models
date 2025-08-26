# USDCOP Trading RL Pipeline

Production-ready reinforcement learning pipeline for USDCOP trading with complete data processing from acquisition to serving.

## Architecture

### Data Pipeline Layers

| Layer | DAG | Purpose | Output |
|-------|-----|---------|--------|
| **L0 - Acquire** | `usdcop_m5__01_l0_acquire` | Data acquisition from MT5/TwelveData | Raw 5-minute bars |
| **L1 - Standardize** | `usdcop_m5__02_l1_standardize` | Standardization and quality checks | Clean OHLCV data |
| **L2 - Prepare** | `usdcop_m5__03_l2_prepare` | Technical indicators and preprocessing | 60+ indicators |
| **L3 - Feature** | `usdcop_m5__04_l3_feature` | Feature engineering and selection | 30 curated features |
| **L4 - RLReady** | `usdcop_m5__05_l4_rlready` | RL environment preparation | Episodes with 17 observations |
| **L5 - Serving** | `usdcop_m5__06_l5_serving_final` | Model training and deployment | ONNX model + serving bundle |

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.9+
- MinIO (S3-compatible storage)
- Apache Airflow

### Installation

```bash
# Clone repository
git clone <repository-url>
cd USDCOP_Trading_RL

# Install dependencies
pip install -r requirements.txt

# Start services
docker-compose up -d
```

### Running the Pipeline

```bash
# Trigger pipeline layers in sequence
airflow dags trigger usdcop_m5__01_l0_acquire_sync_incremental
airflow dags trigger usdcop_m5__02_l1_standardize
airflow dags trigger usdcop_m5__03_l2_prepare
airflow dags trigger usdcop_m5__04_l3_feature
airflow dags trigger usdcop_m5__05_l4_rlready
airflow dags trigger usdcop_m5__06_l5_serving_final
```

### Monitoring
- Airflow UI: http://localhost:8080
- MinIO Console: http://localhost:9001

## Key Features

### L5 Serving Pipeline
- **RL Training**: PPO-LSTM, SAC, DDQN with Stable-Baselines3
- **7 Acceptance Gates**: Comprehensive quality checks
- **ONNX Export**: Optimized model for inference
- **Full Compliance**: Auditor requirements implemented

### Data Quality
- Observation clip rate ≤ 0.5%
- Zero rate < 50% per feature
- Robust median/MAD normalization
- 7-bar global lag for anti-leakage

## Infrastructure

### Docker Services
```yaml
services:
  - airflow-webserver
  - airflow-scheduler
  - airflow-worker
  - postgres
  - redis
  - minio
```

### MinIO Buckets
- `00-l0-ds-usdcop-acquire`
- `01-l1-ds-usdcop-standardize`
- `02-l2-ds-usdcop-prepare`
- `03-l3-ds-usdcop-feature`
- `04-l4-ds-usdcop-rlready`
- `05-l5-ds-usdcop-serving`

## Project Structure

```
USDCOP_Trading_RL/
├── airflow/
│   ├── dags/           # Pipeline DAGs (L0-L5)
│   └── configs/        # YAML configurations
├── src/
│   ├── core/           # Core components
│   ├── models/         # RL models
│   └── trading/        # Trading logic
├── dashboard/          # Web interface
├── docker-compose.yml  # Service orchestration
└── requirements.txt    # Python dependencies
```

## Model Training

L5 trains multiple RL models with 5 seeds each:
- **PPO-LSTM**: Recurrent policy network
- **SAC**: Soft Actor-Critic
- **DDQN**: Double Deep Q-Network

## Deployment

Production serving bundle includes:
- `policy.onnx` - Optimized inference model
- `model_manifest.json` - Metadata and lineage
- `serving_config.json` - Deployment configuration
- L4 contracts for observation processing

## Development

### Testing
```bash
python verify_l5_final.py
```

### Requirements
- Python 3.9+
- Docker 20.10+
- 16GB+ RAM
- 100GB+ storage

## Support

For issues or questions, please contact the development team.

---
**Version**: 2.0.0  
**Status**: Production Ready