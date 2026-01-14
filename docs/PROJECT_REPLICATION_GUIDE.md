# Project Replication Guide

> **Version**: 1.0.0
> **Last Updated**: 2026-01-14
> **Purpose**: Complete instructions to replicate the USDCOP RL Trading System on a new server

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Creating a Backup](#creating-a-backup)
4. [Restoring on New Server](#restoring-on-new-server)
5. [Post-Restore Verification](#post-restore-verification)
6. [Troubleshooting](#troubleshooting)
7. [Component Details](#component-details)

---

## Quick Start

### On Current Server (Create Backup)

```bash
# Navigate to project
cd /path/to/USDCOP-RL-Models

# Run full backup
python scripts/backup/backup_master.py

# Output will be in data/backups/full_backup_YYYYMMDD_HHMMSS/
```

### On New Server (Restore)

```bash
# 1. Clone repository
git clone <repo-url> USDCOP-RL-Models
cd USDCOP-RL-Models

# 2. Copy backup folder to new server
# (Use rsync, scp, or manual copy)

# 3. Run restore
python scripts/backup/restore_master.py --backup-dir /path/to/full_backup_YYYYMMDD_HHMMSS

# 4. Start services
docker-compose up -d
```

---

## Prerequisites

### Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| Docker | 24.0+ | Container runtime |
| Docker Compose | 2.20+ | Service orchestration |
| Python | 3.10+ | Scripts and services |
| Git | 2.40+ | Version control |
| MinIO Client (mc) | Latest | Optional: S3 operations |

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8 cores |
| RAM | 16 GB | 32 GB |
| Disk | 100 GB SSD | 500 GB NVMe |
| Network | 100 Mbps | 1 Gbps |

---

## Creating a Backup

### Full System Backup

```bash
# Complete backup (database + MinIO + Redis + configs)
python scripts/backup/backup_master.py

# Skip MinIO (faster, smaller backup)
python scripts/backup/backup_master.py --skip-minio

# Only configuration files
python scripts/backup/backup_master.py --configs-only
```

### Individual Component Backups

```bash
# Database only
python scripts/backup/backup_database.py --format both

# MinIO only (priority buckets)
python scripts/backup/backup_minio.py --priority-only

# Redis only
python scripts/backup/backup_redis.py
```

### Backup Output Structure

```
data/backups/full_backup_20260114_120000/
├── MASTER_MANIFEST.json       # Backup metadata
├── restore.sh                 # Quick restore script
├── backup_results.json        # Detailed backup results
├── config/                    # Configuration files
│   ├── trading_config.yaml
│   ├── feature_config.json
│   ├── norm_stats.json
│   └── ...
├── models/                    # Model files
│   ├── ppo_production/
│   ├── *.onnx
│   └── *.zip
├── env/                       # Environment files
│   ├── .env
│   └── .env.example
├── init-scripts/              # Database init scripts
│   ├── 00-init-extensions.sql
│   └── ...
├── docker/                    # Docker configuration
│   └── docker-compose.yml
├── database/                  # Database backup
│   └── backup_YYYYMMDD_HHMMSS/
│       ├── schema_ddl.sql.gz
│       ├── usdcop_full_backup_*.sql.gz
│       └── *.csv.gz
├── minio/                     # MinIO backup
│   └── backup_YYYYMMDD_HHMMSS/
│       └── *.tar.gz
└── redis/                     # Redis backup
    └── backup_YYYYMMDD_HHMMSS/
        ├── redis_dump_*.rdb
        └── redis_stream_*.json
```

---

## Restoring on New Server

### Step 1: Prepare New Server

```bash
# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install Python 3.10+
sudo apt install python3.10 python3.10-venv python3-pip

# Install MinIO Client (optional)
wget https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc
sudo mv mc /usr/local/bin/
```

### Step 2: Clone Repository

```bash
git clone <repo-url> USDCOP-RL-Models
cd USDCOP-RL-Models
```

### Step 3: Transfer Backup

```bash
# From source server
rsync -avz --progress data/backups/full_backup_20260114_120000 user@newserver:/path/to/USDCOP-RL-Models/data/backups/

# Or use SCP
scp -r data/backups/full_backup_20260114_120000 user@newserver:/path/to/USDCOP-RL-Models/data/backups/
```

### Step 4: Run Restore

```bash
# Full restore
python scripts/backup/restore_master.py --backup-dir data/backups/full_backup_20260114_120000

# Or use the generated restore script
bash data/backups/full_backup_20260114_120000/restore.sh
```

### Step 5: Start Infrastructure

```bash
# Start core services first
docker-compose up -d postgres redis minio

# Wait for services to be healthy (30-60 seconds)
docker-compose ps

# Initialize MinIO buckets
docker-compose up minio-init
docker wait usdcop-minio-init

# Start remaining services
docker-compose up -d
```

### Step 6: Restore Data (if not done automatically)

```bash
# Restore database
python scripts/backup/restore_master.py --backup-dir data/backups/full_backup_20260114_120000 --database-only

# Restore MinIO
python scripts/backup/restore_master.py --backup-dir data/backups/full_backup_20260114_120000 --minio-only
```

---

## Post-Restore Verification

### Service Health Checks

```bash
# Check all containers are running
docker-compose ps

# Should show all services as "Up" or "healthy"
```

### API Health Endpoints

```bash
# Dashboard
curl http://localhost:5000/api/health

# Trading API
curl http://localhost:8000/health

# Multi-Model API
curl http://localhost:8006/health

# MLOps Inference
curl http://localhost:8090/health
```

### Database Verification

```bash
# Connect to database
docker-compose exec postgres psql -U admin -d usdcop_trading

# Check table counts
SELECT
    schemaname,
    relname as table,
    n_live_tup as rows
FROM pg_stat_user_tables
ORDER BY n_live_tup DESC;

# Check OHLCV data
SELECT COUNT(*) FROM public.usdcop_m5_ohlcv;
SELECT MIN(time), MAX(time) FROM public.usdcop_m5_ohlcv;
```

### MinIO Verification

```bash
# List buckets
mc ls myminio/

# Check model bucket
mc ls myminio/99-common-trading-models/
```

### Web UI Access

| Service | URL | Default Credentials |
|---------|-----|---------------------|
| Dashboard | http://localhost:5000 | N/A |
| Airflow | http://localhost:8080 | From .env |
| MinIO Console | http://localhost:9001 | From .env |
| Grafana | http://localhost:3002 | From .env |
| pgAdmin | http://localhost:5050 | From .env |
| MLflow | http://localhost:5001 | N/A |

---

## Troubleshooting

### Common Issues

#### 1. Docker Compose Fails to Start

```bash
# Check logs
docker-compose logs postgres
docker-compose logs redis

# Restart specific service
docker-compose restart postgres

# Full reset
docker-compose down -v
docker-compose up -d
```

#### 2. Database Connection Refused

```bash
# Verify PostgreSQL is running
docker-compose ps postgres

# Check environment variables
cat .env | grep POSTGRES

# Test connection
docker-compose exec postgres psql -U admin -d usdcop_trading -c "SELECT 1"
```

#### 3. MinIO Buckets Missing

```bash
# Re-run bucket initialization
docker-compose up minio-init

# Or create manually
mc mb myminio/00-raw-usdcop-marketdata
mc mb myminio/99-common-trading-models
# ... (repeat for all buckets)
```

#### 4. Models Not Loading

```bash
# Check model files exist
ls -la models/ppo_production/
ls -la models/*.onnx

# Verify model path in config
cat config/mlops.yaml | grep model_path
```

#### 5. Redis Connection Issues

```bash
# Test Redis
docker-compose exec redis redis-cli PING
# Should return: PONG

# With password
docker-compose exec redis redis-cli -a <password> PING
```

### Log Locations

| Service | Log Location |
|---------|--------------|
| Airflow | `airflow/logs/` |
| API Services | `docker-compose logs <service>` |
| PostgreSQL | `docker-compose logs postgres` |
| Dashboard | `docker-compose logs dashboard` |

---

## Component Details

### Environment Variables (.env)

Critical variables that must be configured:

```bash
# Database
POSTGRES_USER=admin
POSTGRES_PASSWORD=<secure-password>
POSTGRES_DB=usdcop_trading
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# Redis
REDIS_PASSWORD=<secure-password>
REDIS_HOST=redis
REDIS_PORT=6379

# MinIO
MINIO_ACCESS_KEY=<access-key>
MINIO_SECRET_KEY=<secret-key>
MINIO_ENDPOINT=http://minio:9000

# Airflow
AIRFLOW_USER=admin
AIRFLOW_PASSWORD=<secure-password>
AIRFLOW_FERNET_KEY=<generated-key>
AIRFLOW_SECRET_KEY=<generated-key>

# API Keys (24 TwelveData keys)
TWELVEDATA_API_KEY_1=<key>
# ... through TWELVEDATA_API_KEY_8
API_KEY_G1_1=<key>
# ... through API_KEY_G1_8
API_KEY_G2_1=<key>
# ... through API_KEY_G2_8

# Other
FRED_API_KEY=<key>
ANTHROPIC_API_KEY=<key>
DEEPSEEK_API_KEY=<key>
```

### Database Schema

The system uses 7 schemas:

| Schema | Purpose |
|--------|---------|
| public | Core trading data (OHLCV, macro, state) |
| config | Model and feature configuration |
| trading | Model inferences and trades |
| events | Signal event stream |
| metrics | Performance metrics |
| dw | Data warehouse facts |
| staging | ETL staging tables |

### MinIO Buckets

22 buckets for data pipeline:

| Bucket | Purpose |
|--------|---------|
| 00-raw-usdcop-marketdata | Raw OHLCV from MT5/TwelveData |
| 01-l1-ds-usdcop-standardize | UTC-normalized data |
| 02-l2-ds-usdcop-prepare | Cleaned/filtered data |
| 03-l3-ds-usdcop-feature | Technical features |
| 04-l4-ds-usdcop-rlready | Training datasets |
| 05-l5-ds-usdcop-serving | Inference results |
| usdcop-l6-backtest | Backtest results |
| 99-common-trading-models | Trained models |
| 99-common-trading-reports | Performance reports |
| mlflow | Experiment tracking |
| airflow | DAG artifacts |

### Model Files

Required model files for inference:

```
models/
├── ppo_production/
│   ├── model.onnx          # ONNX export
│   ├── config.json         # Model config
│   └── policy_network.pt   # PyTorch weights
├── ppo_v1_*.zip            # SB3 checkpoints
└── onnx/
    └── *.onnx              # Production ONNX models
```

---

## Security Notes

> **WARNING**: The .env file is included for dev/test replication convenience.
> For production deployments:
> 1. Use proper secret management (HashiCorp Vault, AWS Secrets Manager)
> 2. Do NOT commit .env to public repositories
> 3. Rotate all API keys after initial setup
> 4. Enable SSL/TLS for all services
> 5. Configure firewall rules appropriately

---

## Support

For issues or questions:
1. Check `docs/TROUBLESHOOTING.md`
2. Review service logs
3. Open GitHub issue at: https://github.com/anthropics/claude-code/issues

---

*Generated by USDCOP RL Trading System Backup Tools*
