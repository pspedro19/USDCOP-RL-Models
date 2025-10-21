# Frontend API Calls Analysis - USDCOP Trading Dashboard

## Summary
The frontend makes API calls to various endpoints across multiple services. The system uses a mix of:
- Direct REST API calls via `fetch()`
- SWR hooks for data fetching with caching
- Proxy endpoints that forward requests to backend services
- WebSocket connections for real-time updates

---

## 1. API Base URLs & Configuration

### Environment Variables (.env.local)
```
NEXT_PUBLIC_TRADING_API_URL=/api/proxy/trading
NEXT_PUBLIC_WS_URL=ws://48.216.199.139:8082
NEXT_PUBLIC_APP_NAME=USDCOP Trading Dashboard
NEXT_PUBLIC_DEFAULT_SYMBOL=USDCOP
NEXT_PUBLIC_DEFAULT_TIMEFRAME=5m
```

### Internal Service URLs (in code)
```
ANALYTICS_API_URL: process.env.NEXT_PUBLIC_ANALYTICS_API_URL || 'http://localhost:8001'
TRADING_API_URL: 'http://localhost:8000/api' (server-side)
                 '/api/proxy/trading' (client-side)
WS_URL: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8082'
```

---

## 2. Frontend API Endpoints by Category

### A. Market Data Services

#### Real-time Price Data
| Endpoint | Method | Source File | Purpose |
|----------|--------|-------------|---------|
| `/api/proxy/trading/latest/{SYMBOL}` | GET | MarketDataService.ts | Get real-time latest price |
| `/api/proxy/trading/candlesticks/{SYMBOL}` | GET | MarketDataService.ts | Get candlestick/OHLC data |
| `/api/proxy/trading/stats/{SYMBOL}` | GET | MarketDataService.ts | Get symbol statistics |
| `/api/proxy/ws` | GET | MarketDataService.ts, EnhancedDataService.ts | WebSocket proxy (polling fallback) |

**Query Parameters for Candlesticks:**
- `timeframe`: 5m, 15m, 1h, 4h, 1d
- `limit`: 1-10000 (default: 1000)
- `start_date`: ISO date string
- `end_date`: ISO date string
- `include_indicators`: boolean

**Response Data Sent:**
```json
{
  "symbol": "USDCOP",
  "timeframe": "5m",
  "count": 1000,
  "data": [
    {
      "time": 1234567890,
      "open": 4200.5,
      "high": 4205.25,
      "low": 4195.75,
      "close": 4202.5,
      "volume": 1500000,
      "indicators": {
        "ema_20": 4202.1,
        "ema_50": 4201.8,
        "rsi": 55.2,
        "bb_upper": 4210.5,
        "bb_lower": 4195.2
      }
    }
  ]
}
```

#### Historical Data
| Endpoint | Method | Where Called |
|----------|--------|-------------|
| `/api/proxy/trading/api/market/historical` | GET | EnhancedDataService.ts |
| `/api/proxy/trading/api/market/complete-history` | GET | EnhancedDataService.ts |

---

### B. Pipeline Data Endpoints (L0-L6)

These endpoints expose different layers of the data processing pipeline:

| Endpoint | Method | Layer | Purpose | Query Params |
|----------|--------|-------|---------|--------------|
| `/api/pipeline/l0/raw-data` | GET | L0 (Raw) | Raw market OHLC data | limit, offset, start_date, end_date, source |
| `/api/pipeline/l0/statistics` | GET | L0 | Statistics on raw data | |
| `/api/pipeline/l1/episodes` | GET | L1 (Features) | RL episodes data | limit |
| `/api/pipeline/l1/quality-report` | GET | L1 | Data quality metrics | |
| `/api/pipeline/l3/features` | GET | L3 (Correlations) | Feature correlation matrix | limit |
| `/api/pipeline/l4/dataset` | GET | L4 (RL Ready) | Ready-to-train dataset | split (train/test/val) |
| `/api/pipeline/l5/models` | GET | L5 (Model) | ML model artifacts | |
| `/api/pipeline/l6/backtest-results` | GET | L6 (Backtest) | Backtest performance results | split |

**Response Example (L0):**
```json
{
  "data": [
    {
      "timestamp": "2024-01-15T10:30:00Z",
      "symbol": "USDCOP",
      "close": 4285.50,
      "bid": 4285.25,
      "ask": 4285.75,
      "volume": 1200000,
      "source": "postgres"
    }
  ],
  "metadata": {
    "source": "postgres",
    "count": 1000,
    "hasMore": true,
    "table": "market_data"
  }
}
```

---

### C. Trading Signals

| Endpoint | Method | Purpose | Response Contains |
|----------|--------|---------|-------------------|
| `/api/trading/signals` | GET | Get latest trading signals | signals[], performance metrics, technical indicators |
| `/api/trading/signals-test` | GET | Test/demo trading signals | Mock signal data for UI testing |

**Signal Response Structure:**
```json
{
  "success": true,
  "signals": [
    {
      "id": "sig_123456",
      "timestamp": "2024-01-15T10:35:00Z",
      "type": "BUY",
      "confidence": 87.5,
      "price": 4285.50,
      "stopLoss": 4270.00,
      "takeProfit": 4320.00,
      "reasoning": ["RSI oversold", "MACD bullish", "Volume spike"],
      "riskScore": 3.2,
      "expectedReturn": 0.0081,
      "timeHorizon": "15-30 min",
      "modelSource": "L5_PPO_LSTM_v2.1",
      "technicalIndicators": {
        "rsi": 28.5,
        "macd": { "macd": 15.2, "signal": 12.1, "histogram": 3.1 },
        "bollinger": { "upper": 4310, "middle": 4285, "lower": 4260 }
      }
    }
  ],
  "performance": {
    "winRate": 68.5,
    "avgWin": 125.50,
    "profitFactor": 2.34,
    "sharpeRatio": 1.87
  }
}
```

---

### D. ML Analytics Endpoints

| Endpoint | Method | Action Parameter | Purpose | Query Params |
|----------|--------|-------------------|---------|--------------|
| `/api/ml-analytics/models` | GET | list | List available models | action, limit |
| `/api/ml-analytics/models` | GET | metrics | Get model metrics | action, runId |
| `/api/ml-analytics/health` | GET | summary | Model health summary | action |
| `/api/ml-analytics/health` | GET | detail | Detailed model health | action, modelId |
| `/api/ml-analytics/health` | GET | alerts | System alerts | action |
| `/api/ml-analytics/health` | GET | metrics-history | Historical metrics | action, modelId |
| `/api/ml-analytics/health` | POST | N/A | Report health metrics | body: { model_id, metrics } |
| `/api/ml-analytics/predictions` | GET | data | Prediction vs actual data | action, runId, limit, timeRange |
| `/api/ml-analytics/predictions` | GET | metrics | Prediction metrics | action, runId, limit |
| `/api/ml-analytics/predictions` | GET | accuracy-over-time | Accuracy time series | action, runId, limit |
| `/api/ml-analytics/predictions` | GET | feature-impact | Feature importance | action, runId |
| `/api/ml-analytics/predictions` | POST | N/A | Store predictions | body: { predictions[], model_run_id } |

**Prediction Metrics Response:**
```json
{
  "success": true,
  "data": {
    "metrics": {
      "mse": 0.000234,
      "mae": 0.012345,
      "rmse": 0.015301,
      "mape": 0.45,
      "accuracy": 99.55,
      "correlation": 0.9876,
      "total_predictions": 100,
      "correct_direction": 87,
      "direction_accuracy": 87.0
    },
    "sample_predictions": [
      {
        "timestamp": "2024-01-15T10:30:00Z",
        "actual": 4285.50,
        "predicted": 4286.25,
        "confidence": 0.87,
        "percentage_error": 0.0175
      }
    ]
  }
}
```

---

### E. Backtest Endpoints

| Endpoint | Method | Purpose | Response Data |
|----------|--------|---------|----------------|
| `/api/backtest/results` | GET | Get latest backtest results | KPIs, daily returns, trade records |
| `/api/backtest/trigger` | POST | Trigger new backtest run | { forceRebuild: boolean } |

**Backtest Results Response:**
```json
{
  "success": true,
  "data": {
    "runId": "backtest_1234567890",
    "timestamp": "2024-01-15T10:00:00Z",
    "test": {
      "kpis": {
        "top_bar": {
          "CAGR": 0.125,
          "Sharpe": 1.45,
          "Sortino": 1.78,
          "Calmar": 0.89,
          "MaxDD": -0.08,
          "Vol_annualizada": 0.15
        },
        "trading_micro": {
          "win_rate": 0.685,
          "profit_factor": 2.34,
          "payoff": 1.52,
          "expectancy_bps": 145.3
        }
      },
      "dailyReturns": [
        { "date": "2024-01-01", "return": 0.012, "cumulativeReturn": 0.012, "price": 101200 }
      ],
      "trades": [
        {
          "id": "trade_1",
          "timestamp": "2024-01-01T09:30:00Z",
          "symbol": "USDCOP",
          "side": "buy",
          "quantity": 1000,
          "price": 4200,
          "pnl": 500,
          "commission": 0.5
        }
      ]
    }
  }
}
```

---

### F. Health Check Endpoints

| Endpoint | Method | Purpose | Refresh Interval |
|----------|--------|---------|-------------------|
| `/api/market/health` | GET | Check market data API health | On demand |
| `/api/l0/health` | GET | Check L0 pipeline health | On demand |
| `/api/pipeline/health` | GET | Check entire pipeline health | On demand |
| `/api/websocket/status` | GET | Check WebSocket service status | On demand |
| `/api/backup/status` | GET | Check backup system status | On demand |
| `/api/alerts/system` | GET | Get system alerts | On demand |

---

### G. Real-time Market Updates

| Endpoint | Method | Action Param | Purpose |
|----------|--------|--------------|---------|
| `/api/market/realtime` | GET | align | Align real-time data streams |
| `/api/market/realtime` | GET | start | Start real-time data streaming |
| `/api/market/realtime` | GET | stop | Stop real-time data streaming |
| `/api/market/update` | POST | N/A | Post market updates |

---

### H. Analytics Hooks (via Analytics API)

These use SWR for automatic caching and refresh:

| Hook Function | Base URL | Endpoint | Refresh Interval | Query Params |
|--------------|----------|----------|------------------|--------------|
| `useRLMetrics()` | ANALYTICS_API (localhost:8001) | `/api/analytics/rl-metrics` | 60s | symbol, days |
| `usePerformanceKPIs()` | ANALYTICS_API | `/api/analytics/performance-kpis` | 120s | symbol, days |
| `useProductionGates()` | ANALYTICS_API | `/api/analytics/production-gates` | 120s | symbol, days |
| `useRiskMetrics()` | ANALYTICS_API | `/api/analytics/risk-metrics` | 60s | symbol, portfolio_value, days |
| `useSessionPnL()` | ANALYTICS_API | `/api/analytics/session-pnl` | 30s | symbol, session_date |

**Session P&L Response:**
```json
{
  "symbol": "USDCOP",
  "session_date": "2024-01-15",
  "session_open": 4280.00,
  "session_close": 4290.00,
  "session_pnl": 500.50,
  "session_pnl_percent": 0.117,
  "has_data": true,
  "timestamp": "2024-01-15T16:00:00Z"
}
```

---

## 3. Data Flow Diagram

```
Frontend Components
    ↓
Browser-side fetch() / SWR Hooks
    ↓
Next.js API Routes (/api/*)
    ↓
┌─────────────────────────────────┐
│ Backend Services                 │
├─────────────────────────────────┤
│ Port 8000: Trading API           │ → PostgreSQL + MinIO + TwelveData
│ Port 8001: Analytics API         │ → ML Model Results
│ Port 8082: WebSocket Service     │ → Real-time Updates
│ Port 9000: MinIO                 │ → Data Storage
└─────────────────────────────────┘
```

---

## 4. Service-to-Service Communication

### MarketDataService (lib/services/market-data-service.ts)

**Fetch Methods:**
- `connectWebSocket()` → ws://48.216.199.139:8082/ws
- `subscribeToRealTimeUpdates()` → Polls `/api/proxy/ws` every 2 seconds
- `getRealTimeData()` → `/api/proxy/trading/latest/USDCOP`
- `getCandlestickData()` → `/api/proxy/trading/candlesticks/{symbol}`
- `getSymbolStats()` → `/api/proxy/trading/stats/{symbol}`
- `checkAPIHealth()` → `/api/proxy/trading/health`

### EnhancedDataService (lib/services/enhanced-data-service.ts)

**Fetch Methods:**
- `getHistoricalData()` → `/api/proxy/trading/api/market/historical`
- `loadCompleteHistory()` → `/api/proxy/trading/api/market/complete-history`

---

## 5. Component-Level API Usage Examples

### Components Making API Calls:

1. **ConnectionStatus.tsx**
   - GET `/api/market/health` - Check API connectivity

2. **TradingSignals.tsx / SignalAlerts.tsx**
   - GET `/api/trading/signals-test` - Fetch trading signals

3. **ModelPerformanceDashboard.tsx**
   - GET `/api/ml-analytics/models?action=list&limit=10` - List models
   - GET `/api/ml-analytics/health?action=summary` - Health summary
   - GET `/api/ml-analytics/models?action=metrics&runId={id}` - Model metrics
   - GET `/api/ml-analytics/predictions?action=data&runId={id}&limit=50`
   - GET `/api/ml-analytics/predictions?action=accuracy-over-time&runId={id}`
   - GET `/api/ml-analytics/predictions?action=feature-impact&runId={id}`

4. **ModelHealthMonitoring.tsx**
   - GET `/api/ml-analytics/health?action=summary`
   - GET `/api/ml-analytics/health?action=detail&modelId={id}`
   - GET `/api/ml-analytics/health?action=alerts`
   - GET `/api/ml-analytics/health?action=metrics-history&modelId={id}`
   - POST `/api/ml-analytics/health` - Report metrics

5. **Pipeline Dashboard Components (L0-L6)**
   - L0: GET `/api/pipeline/l0/raw-data?limit=1000`
   - L0: GET `/api/pipeline/l0/statistics`
   - L1: GET `/api/pipeline/l1/episodes?limit=100`
   - L3: GET `/api/pipeline/l3/features?limit=100`
   - L4: GET `/api/pipeline/l4/dataset?split=test`
   - L5: GET `/api/pipeline/l5/models`
   - L6: GET `/api/pipeline/l6/backtest-results?split=test`

6. **TradingTerminalView.tsx**
   - GET `/api/data/historical` - Historical data
   - GET `/api/market/realtime?action=align` - Align streams
   - GET `/api/market/realtime?action=start` - Start streaming
   - GET `/api/market/realtime?action=stop` - Stop streaming

7. **RealTimeRiskMonitor.tsx**
   - GET `{TRADING_API_URL}/api/trading/positions?symbol=USDCOP`
   - GET `{ANALYTICS_API_URL}/api/analytics/market-conditions?symbol=USDCOP&days=30`

---

## 6. Authentication & Headers

Most endpoints don't require explicit authentication headers. Default headers used:

```javascript
{
  'Content-Type': 'application/json'
}
```

For POST requests with body:
```javascript
fetch(endpoint, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(data)
})
```

---

## 7. Error Handling Pattern

All services implement fallback strategies:

```
Try Primary API
  ↓
On Error/Failure
  ↓
Try Secondary Source (historical data, mock data, cached data)
  ↓
Return Default/Mock Data
```

Example fallback chain for real-time data:
1. Live WebSocket → `/api/proxy/ws`
2. Real-time API → `/api/proxy/trading/latest/USDCOP`
3. Latest candlestick → `/api/proxy/trading/candlesticks/USDCOP?limit=1`
4. Mock/Cached data

---

## 8. WebSocket Connections

| Protocol | URL | Purpose | Reconnect Policy |
|----------|-----|---------|-------------------|
| WebSocket | ws://48.216.199.139:8082/ws | Real-time price updates | 5s exponential backoff |
| Polling Fallback | GET `/api/proxy/ws` | Alternative if WS blocked | Every 2s |

---

## 9. Query Parameter Standards

### Pagination
- `limit`: Max records to return (1-10000)
- `offset`: Records to skip for pagination

### Date Ranges
- `start_date`: ISO 8601 format (e.g., "2024-01-01")
- `end_date`: ISO 8601 format (e.g., "2024-12-31")

### Filtering
- `symbol`: Trading pair (default: USDCOP)
- `split`: Dataset split (train/test/val)
- `source`: Data source preference (postgres/minio/twelvedata/all)

### Timeframes
- `timeframe`: 5m, 15m, 1h, 4h, 1d

---

## 10. Response Codes Expected

| Code | Meaning | Fallback Action |
|------|---------|-----------------|
| 200 | Success | Use returned data |
| 206 | Partial Content | Use partial data available |
| 404 | Not Found | Try alternative endpoint |
| 425 | Too Early (Market Closed) | Use historical data |
| 500 | Server Error | Use mock/cached data |

---

## Summary Statistics

- **Total Frontend API Endpoints: 45+**
- **Primary Backend Services: 3** (Trading API, Analytics API, WebSocket)
- **Data Pipeline Layers: 7** (L0-L6)
- **Average Refresh Rates: 30s-120s** (varies by data type)
- **Error Handling: 100%** (all services have fallbacks)
- **Maximum Records Per Request: 10,000**

