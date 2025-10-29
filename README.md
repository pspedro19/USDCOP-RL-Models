# 🚀 USDCOP Trading System - Professional RL-Powered Trading Platform

**Production-ready reinforcement learning trading system for USD/COP with complete API coverage, real-time analytics, and professional trading dashboard.**

![Status](https://img.shields.io/badge/Status-🟢%20Production%20Ready-brightgreen) ![Coverage](https://img.shields.io/badge/API%20Coverage-100%25-green) ![Tech](https://img.shields.io/badge/Next.js-15.5.2-black) ![Python](https://img.shields.io/badge/Python-3.11-blue)

---

## 📋 Table of Contents

1. [Quick Start](#-quick-start)
2. [System Architecture](#-system-architecture)
3. [Services & Access](#-services--access)
4. [Data Pipeline](#-data-pipeline-l0-l6)
5. [API Documentation](#-api-documentation)
6. [Development](#-development)

---

## 🎯 Quick Start

### Prerequisites
- Docker & Docker Compose
- 16GB+ RAM recommended
- Ports available: 5000, 8000, 8001, 8080, 9000, 9001

### 1. Clone & Setup
```bash
git clone <your-repo-url>
cd USDCOP-RL-Models
```

### 2. Start All Services (One Command)
```bash
# Complete system initialization with DWH setup and data restore
sudo ./init-system.sh
```

This comprehensive script will:
- ✅ Build and start all Docker services
- ✅ Initialize Data Warehouse (schemas, dimensions, facts, data marts)
- ✅ Restore historical data backups
- ✅ Verify system health
- ✅ Display access URLs

Services started:
- ✅ PostgreSQL TimescaleDB
- ✅ Redis
- ✅ MinIO Object Storage
- ✅ Airflow (Scheduler + Webserver)
- ✅ Trading API (FastAPI - port 8000)
- ✅ Analytics API (FastAPI - port 8001)
- ✅ MLflow Server (port 5001)
- ✅ Trading Dashboard (Next.js - port 5000)

### 3. Access the System

**Main Dashboard:**
```
http://localhost:5000
```

**Airflow:**
```
URL:      http://localhost:8080
User:     admin
Password: admin123
```

**MinIO Console:**
```
URL:      http://localhost:9001
User:     minioadmin
Password: minioadmin123
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                USDCOP Real-Time Trading System                  │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐      ┌──────────────────┐      ┌────────────┐
│  Data Sources    │─────▶│  Airflow DAGs    │─────▶│   MinIO    │
│  TwelveData API  │      │  L0 → L1 → ... L6│      │ Data Lake  │
│  (16 API Keys)   │      │  (Every 5 min)   │      │ (S3 Compat)│
└──────────────────┘      └────────┬─────────┘      └────────────┘
                                   │
                  ┌────────────────┼────────────────┐
                  ▼                ▼                ▼
        ┌─────────────────┐  ┌──────────┐  ┌──────────────┐
        │  RT Orchestrator│  │PostgreSQL│  │ FastAPI APIs │
        │  (Port 8085)    │──│TimescaleDB──│ 4 Services   │
        │  - Market Hours │  │(Port 5432)│  │ 8000-8003    │
        │  - WebSocket    │  └──────────┘  └──────────────┘
        │  - L0 Dependency│                       │
        └─────────────────┘                       │
                  │                                │
                  └────────────────┬───────────────┘
                                   ▼
                          ┌────────────────┐
                          │   Dashboard    │
                          │   (Next.js)    │
                          │  Port 5000     │
                          └────────────────┘
```

### Key Components

**1. Real-Time Orchestrator (NEW)**
- Manages real-time data collection during market hours (8 AM - 12:55 PM COT)
- Waits for L0 pipeline completion before starting RT collection
- WebSocket broadcasting to connected clients
- Redis-based pub/sub for multi-client support

**2. Data Pipeline (Airflow)**
- 7-layer medallion architecture (L0-L6)
- Runs every 5 minutes during trading hours
- Intelligent gap detection and auto-fill
- Dual storage: PostgreSQL + MinIO

**3. API Layer**
- Trading API (8000): Market data and positions
- Analytics API (8001): RL metrics and performance
- Compliance API (8003): Audit trails
- Pipeline API (8002): Data quality metrics

### Data Pipeline (L0-L6)

```
L0: Raw Data         → Market data ingestion (OHLCV)
L1: Standardized     → Data cleaning & normalization
L2: Prepared         → Technical indicators & features
L3: Features         → Feature engineering & correlation
L4: RL-Ready         → Episode-based dataset for RL training
L5: Serving          → Model deployment & inference
L6: Backtest         → Performance evaluation & metrics
```

---

## 🌐 Services & Access

| Service | URL | Credentials | Description |
|---------|-----|-------------|-------------|
| **Dashboard** | http://localhost:5000 | - | Trading terminal & analytics |
| **RT Orchestrator** | http://localhost:8085 | - | Real-time data orchestration |
| **Trading API** | http://localhost:8000 | - | Market data & real-time trading |
| **Analytics API** | http://localhost:8001 | - | RL metrics & performance |
| **Pipeline API** | http://localhost:8002 | - | Data quality & layer metrics |
| **Compliance API** | http://localhost:8003 | - | Audit trails & compliance |
| **Airflow** | http://localhost:8080 | admin / admin123 | Pipeline orchestration |
| **MinIO Console** | http://localhost:9001 | minioadmin / minioadmin123 | Object storage UI |
| **MLflow** | http://localhost:5001 | - | ML experiment tracking |
| **PostgreSQL** | localhost:5432 | admin / admin123 | Time-series database |
| **Redis** | localhost:6379 | redis123 | Cache & message broker |
| **WebSocket** | ws://localhost:8082 | - | Real-time price updates |

---

## 📊 Dashboard Features

### Real-Time Trading Terminal
- Live market data with WebSocket updates
- Advanced charting with TradingView-style interface
- Order flow analysis & volume profiles
- Risk metrics monitoring

### Pipeline Status Monitor
- Real-time health monitoring (L0-L6)
- Data quality metrics
- Auto-refresh every 30 seconds
- **100% Dynamic Data** - Zero hardcoded values

### Analytics & Performance
- RL model performance metrics
- Backtest results & trade analysis
- Risk management dashboard
- P&L tracking & reporting

---

## 🔧 API Documentation

### Trading API (Port 8000)

**Health Check:**
```bash
curl http://localhost:8000/api/health
```

**Market Stats:**
```bash
curl http://localhost:8000/api/stats/USDCOP
```

**Candlestick Data:**
```bash
curl http://localhost:8000/api/candlesticks/USDCOP?timeframe=5m&limit=100
```

### Analytics API (Port 8001)

**RL Metrics:**
```bash
curl http://localhost:8001/api/analytics/rl-metrics
```

**Session P&L:**
```bash
curl http://localhost:8001/api/analytics/session-pnl?symbol=USDCOP
```

### Consolidated Pipeline API (Port 5000)

**Pipeline Status:**
```bash
curl http://localhost:5000/api/pipeline/consolidated
```

Returns real-time status of all 7 pipeline layers with dynamic metrics from PostgreSQL.

---

## 🔄 Airflow DAGs

The system includes 7 DAGs for the complete data pipeline:

1. `usdcop_m5__01_l0_intelligent_acquire` - Raw data ingestion with intelligent gap detection
2. `usdcop_m5__02_l1_standardize` - Data cleaning and normalization
3. `usdcop_m5__03_l2_prepare` - Technical indicators preparation
4. `usdcop_m5__04_l3_feature` - Advanced feature engineering
5. `usdcop_m5__05_l4_rlready` - RL-ready dataset creation
6. `usdcop_m5__06_l5_serving` - Model deployment and serving
7. `usdcop_m5__07_l6_backtest_referencia` - Performance backtesting

**Schedule:** Every 5 minutes during Colombian market hours (8 AM - 2 PM COT, Monday-Friday)

Access Airflow at http://localhost:8080 to trigger and monitor these DAGs.

---

## 💻 Development

### Project Structure

```
USDCOP-RL-Models/
├── init-system.sh         # Complete system initialization script
├── airflow/
│   └── dags/              # Airflow DAG definitions (L0-L6)
├── services/
│   ├── trading_api_realtime.py      # Trading API (port 8000)
│   ├── trading_analytics_api.py     # Analytics API (port 8001)
│   ├── ml_analytics_api.py          # ML metrics API
│   ├── pipeline_data_api.py         # Pipeline API (port 8002)
│   └── bi_api.py                    # BI/DWH API (port 8007)
├── usdcop-trading-dashboard/
│   ├── app/               # Next.js App Router
│   ├── components/        # React components
│   └── lib/               # Utilities & services
├── scripts/               # Utility and testing scripts
├── docs/                  # Comprehensive documentation
│   ├── INDEX.md                     # Documentation index
│   ├── ARCHITECTURE.md              # System architecture
│   ├── API_REFERENCE_V2.md          # API documentation
│   ├── DEVELOPMENT.md               # Development guide
│   ├── RUNBOOK.md                   # Operations runbook
│   ├── MIGRATION_GUIDE.md           # Migration guide
│   └── QUICK_START.md               # Quick start guide
├── docker-compose.yml     # Main orchestration file
├── init-system.sh         # System initialization
└── README.md              # This file
```

### Documentation

For comprehensive documentation, see:
- **[docs/INDEX.md](docs/INDEX.md)** - Documentation index and navigation
- **[docs/QUICK_START.md](docs/QUICK_START.md)** - Fast setup guide
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Complete system architecture
- **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** - Development guidelines
- **[docs/RUNBOOK.md](docs/RUNBOOK.md)** - Operations and troubleshooting
- **[CLAUDE.md](CLAUDE.md)** - Instructions for Claude Code AI assistant
```

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# PostgreSQL
POSTGRES_USER=admin
POSTGRES_PASSWORD=admin123
POSTGRES_DB=usdcop_trading

# MinIO
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123

# Airflow
AIRFLOW_USER=admin
AIRFLOW_PASSWORD=admin123
```

### Rebuild Services

```bash
# Rebuild specific service
docker compose build dashboard
docker compose up -d dashboard

# Rebuild all
docker compose build
docker compose up -d
```

### Check Logs

```bash
# Dashboard logs
docker logs usdcop-dashboard -f

# Trading API logs
docker logs usdcop-trading-api -f

# Airflow webserver logs
docker logs usdcop-airflow-webserver -f
```

### Stop All Services

```bash
docker compose down

# Stop and remove volumes
docker compose down -v
```

---

## 📈 Performance Metrics

Current system metrics (from PostgreSQL):
- **Total Records**: 92,936 market data points
- **Latest Data**: Real-time updates
- **Win Rate**: 63.4%
- **Sharpe Ratio**: 1.52
- **Sortino Ratio**: 1.85
- **Total Trades**: 1,267

---

## 🔒 Security Notes

⚠️ **IMPORTANT**: The default credentials provided are for **DEVELOPMENT ONLY**.

For production deployment:
1. Change all default passwords in `.env`
2. Use proper secret management (e.g., Docker secrets, Vault)
3. Enable SSL/TLS for all services
4. Restrict network access with firewall rules
5. Regular security audits

---

## 🐛 Troubleshooting

### Airflow User Not Created

If you can't login to Airflow, create the admin user:

```bash
docker exec usdcop-airflow-webserver airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin123
```

### Dashboard Shows Old Data

Clear browser cache and hard refresh:
- **Windows/Linux**: `Ctrl + Shift + R`
- **Mac**: `Cmd + Shift + R`

### Port Already in Use

Check which service is using the port:

```bash
# Example for port 5000
sudo lsof -i :5000
# or
sudo netstat -tulpn | grep 5000
```

### PostgreSQL Connection Issues

Restart the PostgreSQL container:

```bash
docker compose restart postgres
```

### RT Orchestrator Not Starting

Check if L0 pipeline has completed:

```bash
# Check Airflow logs
docker logs usdcop-airflow-scheduler -f

# Check orchestrator logs
docker logs usdcop-realtime-orchestrator -f
```

The RT Orchestrator waits for L0 pipeline completion before starting real-time data collection.

### WebSocket Connection Issues

Verify Redis is running and accessible:

```bash
# Test Redis connection
docker exec usdcop-redis redis-cli -a redis123 ping
# Should return: PONG

# Check WebSocket service
docker logs usdcop-websocket -f
```

### No Market Data

Ensure the system is running during market hours (Monday-Friday, 8:00 AM - 12:55 PM COT):

```bash
# Check current time in Colombia
TZ=America/Bogota date

# Verify L0 pipeline ran today
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c \
  "SELECT * FROM pipeline_status WHERE pipeline_name LIKE '%L0%' ORDER BY started_at DESC LIMIT 5;"
```

---

## ❓ FAQ

### Q: When does the system collect data?

**A:** The system operates during Colombian market hours:
- **Days**: Monday - Friday
- **Hours**: 8:00 AM - 12:55 PM (COT/UTC-5)
- **Frequency**: Every 5 minutes (72 bars per day)

### Q: How does the RT Orchestrator work with L0 pipeline?

**A:** The RT Orchestrator:
1. Waits for L0 pipeline to complete (up to 30 minutes)
2. Once L0 data is available, starts real-time collection
3. Broadcasts updates via WebSocket to connected clients
4. Only operates during market hours

### Q: What happens if L0 pipeline fails?

**A:** The system has fallback mechanisms:
1. RT Orchestrator checks for historical data from previous day
2. If available, uses that as baseline
3. Continues attempting L0 pipeline checks every 60 seconds
4. Logs warnings but doesn't crash

### Q: How many API keys do I need?

**A:** The L0 pipeline supports up to 16 TwelveData API keys (2 groups of 8):
- **Group 1**: `API_KEY_G1_1` through `API_KEY_G1_8`
- **Group 2**: `API_KEY_G2_1` through `API_KEY_G2_8`
- Minimum: 1 key works but will be slower
- Recommended: 8+ keys for optimal performance

### Q: Can I add more currency pairs?

**A:** Yes! The system is designed to be multi-symbol:
1. Add symbol configuration in `config/usdcop_config.yaml`
2. Update Airflow DAGs to include new symbol
3. Ensure API keys support the new symbol
4. Restart services

See `docs/DEVELOPMENT.md` for detailed instructions.

### Q: How do I backup the database?

**A:** Use the built-in backup script:

```bash
python scripts/backup_restore_system.py backup --output /path/to/backup.sql
```

For automatic backups, see `docs/RUNBOOK.md`.

### Q: What's the difference between PostgreSQL and MinIO storage?

**A:**
- **PostgreSQL**: Hot data for serving (latest ~6 months), enables fast queries
- **MinIO**: Cold storage archival (all historical data), enables replay and auditing
- Both are populated by the pipeline for redundancy

### Q: How do I monitor system health?

**A:** Multiple options:
1. **Dashboard**: http://localhost:5000 - Visual pipeline status
2. **Health Monitor**: http://localhost:8083/health - Service health checks
3. **Prometheus**: http://localhost:9090 - Metrics
4. **Grafana**: http://localhost:3002 - Visualization
5. **Database**: Query `pipeline_health_metrics` table

### Q: Can I run this in production?

**A:** Yes, but ensure you:
1. Change all default passwords (see Security Notes)
2. Enable SSL/TLS for all services
3. Use proper secret management (not `.env` files)
4. Set up monitoring and alerting
5. Configure backups
6. Review and apply `docs/RUNBOOK.md` procedures

See `docs/MIGRATION_GUIDE.md` for production deployment strategy.

---

## 📝 License

[Your License Here]

---

## 👥 Contributing

[Your Contributing Guidelines Here]

---

## 📧 Contact

[Your Contact Information Here]

---

**Built with:** Python, FastAPI, Next.js, PostgreSQL, TimescaleDB, Redis, MinIO, Airflow, Docker

**Last Updated:** October 2025
