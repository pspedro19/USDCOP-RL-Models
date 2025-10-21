# Frontend API Quick Reference

## All Frontend API Endpoints (Sorted by Category)

### Market Data Endpoints
- **GET** `/api/proxy/trading/latest/{SYMBOL}` - Real-time price
- **GET** `/api/proxy/trading/candlesticks/{SYMBOL}` - OHLC data (timeframe, limit, start_date, end_date)
- **GET** `/api/proxy/trading/stats/{SYMBOL}` - Symbol statistics
- **GET** `/api/proxy/ws` - WebSocket proxy/polling fallback
- **GET** `/api/proxy/trading/api/market/historical` - Historical data
- **GET** `/api/proxy/trading/api/market/complete-history` - Complete history

### Pipeline Data (L0-L6)
- **GET** `/api/pipeline/l0/raw-data` - Raw OHLC data (limit, offset, start_date, end_date, source)
- **GET** `/api/pipeline/l0/statistics` - L0 statistics
- **GET** `/api/pipeline/l0` - L0 overview
- **GET** `/api/pipeline/l1/episodes` - RL episodes (limit)
- **GET** `/api/pipeline/l1/quality-report` - Data quality
- **GET** `/api/pipeline/l3/features` - Features correlation (limit)
- **GET** `/api/pipeline/l4/dataset` - RL dataset (split: train/test/val)
- **GET** `/api/pipeline/l5/models` - Model artifacts
- **GET** `/api/pipeline/l6/backtest-results` - Backtest results (split)

### Trading Signals
- **GET** `/api/trading/signals` - Live trading signals
- **GET** `/api/trading/signals-test` - Test signals

### ML Analytics
- **GET** `/api/ml-analytics/models?action=list` - List models
- **GET** `/api/ml-analytics/models?action=metrics&runId=X` - Model metrics
- **GET** `/api/ml-analytics/health?action=summary` - Health summary
- **GET** `/api/ml-analytics/health?action=detail&modelId=X` - Model detail
- **GET** `/api/ml-analytics/health?action=alerts` - Alerts
- **GET** `/api/ml-analytics/health?action=metrics-history&modelId=X` - Metrics history
- **POST** `/api/ml-analytics/health` - Report health
- **GET** `/api/ml-analytics/predictions?action=data` - Predictions vs actuals
- **GET** `/api/ml-analytics/predictions?action=metrics` - Metrics
- **GET** `/api/ml-analytics/predictions?action=accuracy-over-time` - Accuracy timeline
- **GET** `/api/ml-analytics/predictions?action=feature-impact` - Feature importance
- **POST** `/api/ml-analytics/predictions` - Store predictions

### Backtest
- **GET** `/api/backtest/results` - Latest backtest results
- **POST** `/api/backtest/trigger` - Trigger backtest

### Health & Status
- **GET** `/api/market/health` - Market API health
- **GET** `/api/l0/health` - L0 health
- **GET** `/api/pipeline/health` - Pipeline health
- **GET** `/api/websocket/status` - WebSocket status
- **GET** `/api/backup/status` - Backup status
- **GET** `/api/alerts/system` - System alerts

### Real-time Market
- **GET** `/api/market/realtime?action=align` - Align streams
- **GET** `/api/market/realtime?action=start` - Start streaming
- **GET** `/api/market/realtime?action=stop` - Stop streaming
- **POST** `/api/market/update` - Post market update

### Analytics API (External Service)
- **GET** `{ANALYTICS_API}/api/analytics/rl-metrics` - RL metrics (symbol, days)
- **GET** `{ANALYTICS_API}/api/analytics/performance-kpis` - Performance KPIs
- **GET** `{ANALYTICS_API}/api/analytics/production-gates` - Production gates
- **GET** `{ANALYTICS_API}/api/analytics/risk-metrics` - Risk metrics
- **GET** `{ANALYTICS_API}/api/analytics/session-pnl` - Session P&L
- **GET** `{ANALYTICS_API}/api/analytics/market-conditions` - Market conditions

---

## HTTP Methods Used
- **GET** - Fetch data (45+ endpoints)
- **POST** - Submit/trigger operations (4 endpoints)
- **WebSocket (ws://)** - Real-time connections

---

## Common Query Parameters

### Pagination
- `limit` - Max records (1-10000)
- `offset` - Skip records

### Filtering
- `symbol` - Trading pair (default: USDCOP)
- `split` - train/test/val
- `source` - postgres/minio/twelvedata/all
- `action` - Various actions (list, metrics, summary, etc.)

### Date Range
- `start_date` - ISO date (2024-01-01)
- `end_date` - ISO date (2024-12-31)

### Timeframe
- `timeframe` - 5m, 15m, 1h, 4h, 1d

---

## Data Sent to Backend (Request Body)

### POST /api/backtest/trigger
```json
{ "forceRebuild": boolean }
```

### POST /api/ml-analytics/health
```json
{ "model_id": string, "metrics": object }
```

### POST /api/ml-analytics/predictions
```json
{ "predictions": array, "model_run_id": string }
```

### POST /api/market/update
```json
{ /* market update data */ }
```

---

## Backend Services Called

| Service | Port | Role |
|---------|------|------|
| Trading API | 8000 | Market data, OHLC, real-time prices |
| Analytics API | 8001 | ML metrics, risk, performance |
| WebSocket | 8082 | Real-time data streaming |
| MinIO | 9000 | Object storage for raw data |
| PostgreSQL | 5432 | Timescale DB (92K+ market records) |

---

## Key Files

### Service Implementations
- `/lib/services/market-data-service.ts` - Market data fetching
- `/lib/services/enhanced-data-service.ts` - Enhanced data access
- `/lib/services/backtest-client.ts` - Backtest operations
- `/lib/services/pipeline-data-client.ts` - Pipeline data

### Hooks (Auto-refresh with SWR)
- `/hooks/useAnalytics.ts` - Analytics hooks (RL metrics, KPIs, risk, etc.)
- `/hooks/useMarketStats.ts` - Market stats hook

### Components Making Calls
- Components in `/components/ml-analytics/`
- Components in `/components/views/`
- Components in `/components/trading/`
- Components in `/components/status/`

### API Routes
- All routes in `/app/api/` (proxy and handlers)

---

## Response Codes

| Code | Meaning | Handled By |
|------|---------|-----------|
| 200 | Success | Return data |
| 206 | Partial | Use partial data |
| 404 | Not found | Try alternative |
| 425 | Market closed | Use historical |
| 500 | Server error | Use mock/fallback |

---

## Error Handling Strategy

All services implement multi-level fallbacks:

1. Try primary API
2. Try secondary source
3. Try historical data
4. Use cached data
5. Use mock data
6. Return error state

Example: Real-time data
- WebSocket (live) → REST API → Candlestick API → Historical → Mock

---

## Refresh Intervals

- Session P&L: 30 seconds
- Risk Metrics: 60 seconds
- Performance KPIs: 120 seconds
- Production Gates: 120 seconds
- Market Stats: 30 seconds (default)
- Health Checks: On demand
- Trading Signals: On demand
- Pipeline Data: On demand

---

## Authentication

Most endpoints are open (no auth required). Headers used:
```javascript
{
  'Content-Type': 'application/json'
}
```

---

## Total Counts

- **Endpoints**: 45+
- **HTTP Methods**: GET (41), POST (4)
- **WebSocket Connections**: 1
- **Service Layers**: 7 (L0-L6 pipeline)
- **Backend Services**: 3 main (Trading, Analytics, WebSocket)

