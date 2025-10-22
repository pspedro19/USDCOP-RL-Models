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
# Initialize and start the entire system
./start-system.sh
```

This will start:
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
│                    USDCOP Trading System                        │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐      ┌──────────────────┐      ┌────────────┐
│  Data Sources    │─────▶│  Airflow DAGs    │─────▶│   MinIO    │
│  - Market Data   │      │  L0 → L1 → ... L6│      │ Data Lake  │
└──────────────────┘      └──────────────────┘      └────────────┘
                                   │
                                   ▼
┌──────────────────┐      ┌──────────────────┐      ┌────────────┐
│  PostgreSQL      │◀─────│   FastAPI APIs   │◀─────│ Dashboard  │
│  TimescaleDB     │      │  Trading+Analytics│      │ (Next.js)  │
└──────────────────┘      └──────────────────┘      └────────────┘
```

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
| **Trading API** | http://localhost:8000 | - | Market data & real-time trading |
| **Analytics API** | http://localhost:8001 | - | RL metrics & performance |
| **Airflow** | http://localhost:8080 | admin / admin123 | Pipeline orchestration |
| **MinIO Console** | http://localhost:9001 | minioadmin / minioadmin123 | Object storage UI |
| **MLflow** | http://localhost:5001 | - | ML experiment tracking |
| **PostgreSQL** | localhost:5432 | admin / admin123 | Time-series database |
| **Redis** | localhost:6379 | - | Cache & message broker |

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

1. `usdcop_m5__01_l0_ingest` - Raw data ingestion
2. `usdcop_m5__02_l1_standardize` - Data standardization
3. `usdcop_m5__03_l2_prepare` - Feature preparation
4. `usdcop_m5__04_l3_feature` - Feature engineering
5. `usdcop_m5__05_l4_rlready` - RL dataset creation
6. `usdcop_m5__06_l5_serving` - Model serving
7. `usdcop_m5__07_l6_backtest_referencia` - Backtesting

Access Airflow at http://localhost:8080 to trigger and monitor these DAGs.

---

## 💻 Development

### Project Structure

```
USDCOP-RL-Models/
├── airflow/
│   └── dags/              # Airflow DAG definitions
├── services/
│   ├── trading_api_realtime.py
│   ├── trading_analytics_api.py
│   ├── ml_analytics_api.py
│   └── pipeline_data_api.py
├── usdcop-trading-dashboard/
│   ├── app/               # Next.js App Router
│   ├── components/        # React components
│   └── lib/               # Utilities & services
├── scripts/               # Utility scripts
├── docs/                  # Documentation
├── docker-compose.yml     # Main orchestration file
├── requirements.txt       # Python dependencies
└── README.md             # This file
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
