# USD/COP Trading Platform - Deployment Guide

## Quick Deployment (5 minutes)

### Prerequisites
- Docker & Docker Compose installed
- Git access to repository
- TwelveData API key (free tier works)

### Steps

```bash
# 1. Clone repository
git clone https://github.com/your-org/USDCOP-RL-Models.git
cd USDCOP-RL-Models

# 2. Configure environment
cp .env.example .env
# Edit .env with your credentials (minimum: TWELVEDATA_API_KEY_1, POSTGRES_PASSWORD)

# 3. Start services
docker-compose up -d

# 4. Wait for services to initialize (~2 minutes)
docker-compose ps

# 5. Restore platform data
./scripts/restore_platform.sh
```

### Access Points
| Service | URL | Default Credentials |
|---------|-----|---------------------|
| Dashboard | http://localhost:3000 | - |
| Airflow | http://localhost:8080 | admin/admin |
| pgAdmin | http://localhost:5050 | See .env |
| Grafana | http://localhost:3001 | admin/admin |

---

## What Gets Restored

### Database (usdcop_backup.sql.gz - 3.9 MB)
- `usdcop_m5_ohlcv`: ~15,000+ OHLCV bars (Oct 2024 - Jan 2026)
- `config.models`: Model registry (V19, V20)
- `trading.model_inferences`: Historical inference records
- `trading.paper_positions`: Paper trading positions
- All schemas: public, config, trading, macro

### ML Models (~2 MB total)
- `ppo_v1_20251226_054154.zip` - V19 Original (1.8 MB)
- `ppo_v20_production/final_model.zip` - V20 Macro (154 KB)
- `ppo_v20_production/model_v20.onnx` - ONNX format (22 KB)

### Airflow DAGs
- `v3.l0_ohlcv_realtime` - OHLCV capture (every 5 min)
- `v3.l0_macro_unified` - Macro data scraping (daily)
- `v3.l5_multi_model_inference` - RL inference (every 5 min)

---

## Market Hours

**Trading Window**: Monday-Friday 8:00-12:55 COT (Colombia Time)
- UTC equivalent: 13:00-17:55 UTC
- Airflow schedule: `*/5 13-17 * * 1-5`

**No Trading**:
- Weekends
- Colombian holidays (automatic detection via TradingCalendar)

---

## Environment Variables (Minimum Required)

```bash
# Database
POSTGRES_PASSWORD=your_secure_password

# API Keys
TWELVEDATA_API_KEY_1=your_twelvedata_key

# Optional but recommended
REDIS_PASSWORD=your_redis_password
```

---

## Troubleshooting

### Services not starting
```bash
# Check logs
docker-compose logs -f

# Restart specific service
docker-compose restart trading-api
```

### Database connection errors
```bash
# Verify PostgreSQL is running
docker exec usdcop-timescaledb pg_isready -U trading_user

# Check connection
docker exec -it usdcop-timescaledb psql -U trading_user -d trading_db -c "SELECT NOW()"
```

### Models not loading
```bash
# Verify models exist in container
docker exec usdcop-airflow ls -la /opt/airflow/ml_models/

# Re-run restore script
./scripts/restore_platform.sh
```

### Chart showing wrong timezone
The chart auto-detects timestamp format:
- OLD format (13:00-17:55 UTC) - converts to COT
- NEW format (08:00-12:55 COT) - displays as-is

---

## Production Checklist

- [ ] Strong passwords in .env (use `openssl rand -base64 32`)
- [ ] Valid TwelveData API key
- [ ] Database backup restored
- [ ] Models copied to container
- [ ] Airflow DAGs enabled
- [ ] Dashboard accessible
- [ ] Signals appearing during market hours

---

## Files Structure

```
USDCOP-RL-Models/
├── .env.example          # Environment template
├── docker-compose.yml    # Service definitions
├── data/
│   └── backups/
│       └── usdcop_backup.sql.gz  # Full DB backup
├── models/
│   ├── ppo_v1_20251226_054154.zip     # V19 model
│   └── ppo_v20_production/
│       ├── final_model.zip             # V20 model
│       └── model_v20.onnx              # ONNX version
├── scripts/
│   └── restore_platform.sh  # Auto-restoration
├── airflow/dags/
│   ├── l0_ohlcv_realtime.py
│   ├── l0_macro_unified.py
│   └── l5_multi_model_inference.py
└── usdcop-trading-dashboard/
    └── ...                  # Next.js dashboard
```

---

## Support

- Issues: https://github.com/your-org/USDCOP-RL-Models/issues
- Author: Pedro @ Lean Tech Solutions
- Version: 3.0.0
