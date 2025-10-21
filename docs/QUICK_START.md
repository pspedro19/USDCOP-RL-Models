# 🚀 Quick Start Guide - USDCOP Trading System

## ⚡ 5-Minute Setup

### Step 1: Clone Repository
```bash
git clone https://github.com/pspedro19/USDCOP-RL-Models.git
cd USDCOP-RL-Models
```

### Step 2: Start Backend Services
```bash
# Make scripts executable
chmod +x start-all-apis.sh check-api-status.sh stop-all-apis.sh

# Start all 7 API services
./start-all-apis.sh

# Wait ~10 seconds, then check status
./check-api-status.sh
```

**Expected Output:**
```
✓ Trading API (8000):        ● Running
✓ Analytics API (8001):      ● Running
✓ Trading Signals API (8003): ● Running
✓ Pipeline Data API (8004):  ● Running
✓ ML Analytics API (8005):   ● Running
✓ Backtest API (8006):       ● Running
✓ WebSocket Service (8082):  ● Running

Total: 7/7 services running ✅
```

### Step 3: Start Dashboard
```bash
cd usdcop-trading-dashboard
npm install  # First time only
npm run dev
```

### Step 4: Access Dashboard
```
Open browser: http://localhost:3001
Login: admin / admin
```

---

## 🎯 First Steps After Login

### 1. Trading Dashboard (Home View)
- See real-time USD/COP price
- View candlestick chart
- Monitor 24h statistics

### 2. Trading Signals
- Click "Trading Signals" in sidebar
- See ML-powered BUY/SELL/HOLD recommendations
- Review confidence scores and technical indicators

### 3. Executive Overview
- Click "Executive Overview"
- Check KPIs: Sortino, Sharpe, Calmar ratios
- Verify production gates status

### 4. Risk Monitor
- Click "Risk Monitor"
- Review VaR, Max Drawdown
- Check position exposure

---

## 🔧 Common Commands

### Check Service Status
```bash
./check-api-status.sh
```

### View Logs
```bash
# All services
tail -f logs/api/*.log

# Specific service
tail -f logs/api/Trading-Signals-API.log
```

### Stop All Services
```bash
./stop-all-apis.sh
```

### Restart Service
```bash
# Stop
./stop-all-apis.sh

# Start
./start-all-apis.sh
```

---

## 📊 Test API Endpoints

### Get Latest Price
```bash
curl http://localhost:8000/api/latest/USDCOP
```

### Get Trading Signals
```bash
curl http://localhost:8003/api/trading/signals
```

### Get ML Models
```bash
curl http://localhost:8005/api/ml-analytics/models?action=list
```

### Get Backtest Results
```bash
curl http://localhost:8006/api/backtest/results
```

---

## 🐛 Troubleshooting

### Service Won't Start

**Problem:** Port already in use
```bash
# Find process using port 8000
lsof -i :8000

# Kill it
kill -9 <PID>

# Restart
./start-all-apis.sh
```

### Dashboard Shows No Data

**Check:**
1. Backend services running: `./check-api-status.sh`
2. Check browser console for errors (F12)
3. Verify API endpoints accessible:
   ```bash
   curl http://localhost:8000/api/latest/USDCOP
   ```

### Database Connection Error

```bash
# Check PostgreSQL running
docker ps | grep postgres

# If not running, start Docker services
docker-compose up -d
```

---

## 📚 Next Steps

1. **Read Documentation**
   - `docs/API_REFERENCE.md` - All API endpoints
   - `docs/DASHBOARD_VIEWS.md` - Dashboard guide
   - `docs/ENDPOINT_COVERAGE.md` - Coverage matrix

2. **Explore Dashboard**
   - Try all 13 professional views
   - Review charts and indicators
   - Check risk management tools

3. **Test APIs**
   - Visit http://localhost:8000/docs for Swagger UI
   - Test endpoints with curl or Postman

4. **Configure System**
   - Set environment variables
   - Customize dashboard settings
   - Configure alert thresholds

---

## 🌐 Access Points

| Service | URL | Credentials |
|---------|-----|-------------|
| **Dashboard** | http://localhost:3001 | admin / admin |
| **Trading API Docs** | http://localhost:8000/docs | - |
| **Signals API** | http://localhost:8003/api/trading/signals | - |
| **ML Analytics** | http://localhost:8005/api/ml-analytics/models | - |
| **Backtest API** | http://localhost:8006/api/backtest/status | - |

---

## ✅ System Health Check

```bash
# Quick health check script
curl http://localhost:8000/api/market/health
curl http://localhost:8001/api/health
curl http://localhost:8003/api/health
curl http://localhost:8004/api/health
curl http://localhost:8005/api/ml-analytics/models?action=list
curl http://localhost:8006/api/backtest/status
```

All should return `200 OK` with JSON response.

---

**Need Help?** See main `README.md` or check `docs/` folder for detailed documentation.
