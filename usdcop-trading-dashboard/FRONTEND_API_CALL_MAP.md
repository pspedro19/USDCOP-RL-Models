# Frontend API Call Map - Visual Reference

## Component → API Call Relationships

### Dashboard Views

```
┌─ L0RawDataDashboard.tsx
│  ├─ GET /api/pipeline/l0/raw-data?limit=1000
│  └─ GET /api/pipeline/l0/statistics
│
├─ L1FeatureStats.tsx
│  └─ GET /api/pipeline/l1/episodes?limit=100
│
├─ L3Correlations.tsx
│  └─ GET /api/pipeline/l3/features?limit=100
│
├─ L4RLReadyData.tsx
│  └─ GET /api/pipeline/l4/dataset?split=test
│
├─ L5ModelDashboard.tsx
│  └─ GET /api/pipeline/l5/models
│
└─ L6BacktestResults.tsx
   └─ GET /api/pipeline/l6/backtest-results?split=test
```

### ML Analytics Components

```
┌─ ModelPerformanceDashboard.tsx
│  ├─ GET /api/ml-analytics/models?action=list&limit=10
│  ├─ GET /api/ml-analytics/health?action=summary
│  ├─ GET /api/ml-analytics/models?action=metrics&runId={runId}
│  ├─ GET /api/ml-analytics/predictions?action=data&runId={runId}&limit=50
│  ├─ GET /api/ml-analytics/predictions?action=accuracy-over-time&runId={runId}&limit=100
│  └─ GET /api/ml-analytics/predictions?action=feature-impact&runId={runId}
│
├─ ModelHealthMonitoring.tsx
│  ├─ GET /api/ml-analytics/health?action=summary
│  ├─ GET /api/ml-analytics/health?action=detail&modelId={modelId}
│  ├─ GET /api/ml-analytics/health?action=alerts
│  ├─ GET /api/ml-analytics/health?action=metrics-history&modelId={modelId}
│  └─ POST /api/ml-analytics/health
│
├─ FeatureImportanceChart.tsx
│  └─ GET /api/ml-analytics/predictions?action=feature-impact&runId={runId}
│
└─ PredictionsVsActualsChart.tsx
   ├─ GET /api/ml-analytics/predictions?action=data&runId={runId}&limit={limit}&timeRange={timeRange}
   └─ GET /api/ml-analytics/predictions?action=metrics&runId={runId}&limit={limit}
```

### Trading Components

```
┌─ TradingSignals.tsx / SignalAlerts.tsx
│  └─ GET /api/trading/signals-test
│
├─ TradingTerminalView.tsx
│  ├─ GET /api/data/historical
│  ├─ GET /api/market/realtime?action=align
│  ├─ GET /api/market/realtime?action=start
│  └─ GET /api/market/realtime?action=stop
│
└─ RealTimeRiskMonitor.tsx
   ├─ GET {TRADING_API_URL}/api/trading/positions?symbol=USDCOP
   └─ GET {ANALYTICS_API_URL}/api/analytics/market-conditions?symbol=USDCOP&days=30
```

### Status Components

```
└─ ConnectionStatus.tsx
   └─ GET /api/market/health
```

---

## Service Classes → API Call Flow

### MarketDataService

```
User Component
    ↓
useMarketStats() Hook
    ↓
MarketDataService.getSymbolStats(symbol)
    ↓
    ├─ Try: GET /api/proxy/trading/stats/{symbol}
    │   ├─ Success → Return stats
    │   └─ Fail
    │       ↓
    │       └─ Fallback: getCandlestickData()
    │
    └─ getCandlestickData()
        ├─ GET /api/proxy/trading/candlesticks/{symbol}?timeframe=5m&limit=288
        └─ Calculate stats from response
```

### Real-time Price Updates

```
MarketDataService.subscribeToRealTimeUpdates()
    ↓
    ├─ Connect WebSocket: ws://48.216.199.139:8082/ws
    │   ├─ Send: { type: 'subscribe', symbol: 'USDCOP' }
    │   └─ Receive: price updates every tick
    │
    └─ Fallback (if WS blocked):
        ├─ Poll: GET /api/proxy/ws (every 2 seconds)
        ├─ Try: GET /api/proxy/trading/latest/USDCOP
        ├─ Try: GET /api/proxy/trading/candlesticks/USDCOP?limit=1
        └─ Use: Cached/Mock data
```

### Candlestick Data

```
Component needs OHLC data
    ↓
MarketDataService.getCandlestickData(symbol, timeframe, startDate, endDate, limit)
    ↓
GET /api/proxy/trading/candlesticks/{symbol}?
    timeframe=5m
    &limit=1000
    &start_date=2024-01-01
    &end_date=2024-12-31
    &include_indicators=true
    ↓
Response:
{
  symbol: "USDCOP",
  timeframe: "5m",
  count: 1000,
  data: [
    { time, open, high, low, close, volume, indicators: {...} }
  ]
}
```

---

## Hook-Based API Calls (Auto-refresh)

### useAnalytics.ts

```
Component
    ↓
useAnalytics Hook
    ├─ useRLMetrics()
    │  ├─ GET {ANALYTICS_API}/api/analytics/rl-metrics?symbol=USDCOP&days=30
    │  └─ Refresh every 60 seconds
    │
    ├─ usePerformanceKPIs()
    │  ├─ GET {ANALYTICS_API}/api/analytics/performance-kpis?symbol=USDCOP&days=90
    │  └─ Refresh every 120 seconds
    │
    ├─ useProductionGates()
    │  ├─ GET {ANALYTICS_API}/api/analytics/production-gates?symbol=USDCOP&days=90
    │  └─ Refresh every 120 seconds
    │
    ├─ useRiskMetrics()
    │  ├─ GET {ANALYTICS_API}/api/analytics/risk-metrics?symbol=USDCOP&portfolio_value=10000000&days=30
    │  └─ Refresh every 60 seconds
    │
    └─ useSessionPnL()
       ├─ GET {ANALYTICS_API}/api/analytics/session-pnl?symbol=USDCOP
       └─ Refresh every 30 seconds
```

### useMarketStats.ts

```
Component
    ↓
useMarketStats(symbol, refreshInterval=30000)
    ├─ Initial: MarketDataService.getSymbolStats(symbol)
    ├─ Auto-refresh every 30 seconds
    ├─ Health check: MarketDataService.checkAPIHealth()
    │  └─ GET /api/proxy/trading/health
    └─ Session P&L: fetch(`{ANALYTICS_API}/api/analytics/session-pnl?symbol=${symbol}`)
```

---

## Data Pipeline Access Pattern

```
Dashboard Component (e.g., L0RawDataDashboard)
    ↓
useEffect / onClick
    ↓
fetch('/api/pipeline/l0/raw-data?limit=1000&offset=0&source=postgres')
    ↓
Backend API Route Handler (/app/api/pipeline/l0/raw-data/route.ts)
    ↓
Data Sources (Priority Order):
    1. PostgreSQL (Primary)
       └─ SELECT FROM market_data WHERE symbol='USDCOP'
    2. MinIO (Archive)
       └─ Bucket: 00-raw-usdcop-marketdata
    3. TwelveData API (Real-time)
       └─ External API call
    ↓
Response:
{
  data: [{ timestamp, symbol, close, bid, ask, volume, source }],
  metadata: { source: 'postgres', count: 1000, hasMore: true }
}
```

---

## ML Predictions Workflow

```
Component: ModelPerformanceDashboard
    ↓
fetch('/api/ml-analytics/predictions?action=data&runId=latest&limit=50')
    ↓
Response:
{
  data: [
    {
      timestamp: "2024-01-15T10:30:00Z",
      actual: 4285.50,
      predicted: 4286.25,
      confidence: 0.87,
      feature_values: { rsi: 45.2, macd: 12.1, ... },
      error: 0.75,
      percentage_error: 0.0175
    }
  ]
}
    ↓
Metrics Calculation:
- MSE, MAE, RMSE, MAPE
- Accuracy, Correlation
- Direction Accuracy
```

---

## Backtest Workflow

```
User Clicks "Trigger Backtest"
    ↓
backtestClient.triggerBacktest(forceRebuild=true)
    ↓
POST /api/backtest/trigger
{ forceRebuild: true }
    ↓
Backend starts backtest process
    ↓
Component polls: GET /api/backtest/results
    ↓
Response:
{
  runId: "backtest_1234567890",
  timestamp: "2024-01-15T10:00:00Z",
  test: { kpis: {...}, dailyReturns: [...], trades: [...] },
  val: { kpis: {...}, dailyReturns: [...], trades: [...] }
}
    ↓
Display Results
```

---

## Error Handling Flow

```
fetch('/api/endpoint')
    ↓
┌─ Success (200) ────→ Return Data
├─ Partial (206) ────→ Use Available Data
├─ Market Closed (425) ──→ Use Historical Fallback
├─ Not Found (404) ──→ Try Alternative Endpoint
└─ Error (500) ──→ Use Mock/Cached Data
```

---

## Cross-Service Communication

```
Frontend
    │
    ├─ fetch('/api/proxy/trading/*')
    │   ↓
    │   Next.js Proxy Route
    │   ↓
    │   Trading API (localhost:8000)
    │   ├─ Database: PostgreSQL
    │   ├─ Cache: MinIO (9000)
    │   └─ External: TwelveData API
    │
    ├─ fetch('/api/ml-analytics/*')
    │   ↓
    │   Next.js Handler Route
    │   ↓
    │   Analytics API (localhost:8001)
    │   └─ ML Model Results
    │
    ├─ WebSocket Connection
    │   ↓
    │   ws://48.216.199.139:8082
    │   └─ Real-time Price Ticks
    │
    └─ fetch('/api/pipeline/*')
        ↓
        Next.js Handler Route
        ↓
        PostgreSQL / MinIO / TwelveData
```

---

## Request Patterns

### GET Requests (Most Common)
```
GET /api/pipeline/l0/raw-data?limit=100&offset=0&start_date=2024-01-01
Accept: application/json
```

### POST Requests (Action/State Change)
```
POST /api/backtest/trigger
Content-Type: application/json

{ "forceRebuild": true }
```

### WebSocket
```
Initial Connection: ws://48.216.199.139:8082/ws
Send: { "type": "subscribe", "symbol": "USDCOP" }
Receive: { "type": "price_update", "symbol": "USDCOP", "price": 4285.50, ... }
```

---

## Performance Characteristics

| Endpoint Type | Typical Latency | Cache Duration | Refresh Interval |
|---------------|-----------------|-----------------|------------------|
| Real-time (WS) | <100ms | N/A | Live/Tick |
| Market Data | 100-500ms | 2s-30s | 30s-120s |
| ML Analytics | 200-1000ms | 1min | 60s-120s |
| Pipeline Data | 500-2000ms | 5min-1hr | On demand |
| Health Checks | 100-200ms | 1min | On demand |

