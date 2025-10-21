# API Documentation Index

Complete analysis of all frontend API calls made by the USDCOP Trading Dashboard.

## Documentation Files (996 lines total)

### 1. README_API_ANALYSIS.md (277 lines)
**Start here** - Overview and guide to all documentation

Contains:
- Quick facts about the API system
- API categories overview
- Key service files
- Common usage patterns
- Environment configuration
- Data flow examples
- Refresh intervals
- Error handling chains
- Performance tips
- Troubleshooting guide

### 2. API_CALLS_ANALYSIS.md (503 lines)
**Comprehensive reference** - Detailed technical documentation

Contains:
- All API endpoints grouped by category
- Query parameters and response formats
- Service-to-service communication patterns
- Component usage examples
- Authentication details
- Response codes and meanings
- WebSocket connections
- Full data flow diagrams
- Summary statistics

### 3. API_QUICK_REFERENCE.md (216 lines)
**Quick lookup** - Fast reference guide

Contains:
- All 45+ endpoints sorted by category
- HTTP methods (GET 41, POST 4, WebSocket 1)
- Common query parameters
- Request/response body examples
- Backend services and ports
- Key file locations
- Refresh intervals table
- Response codes table
- Total counts summary

### 4. FRONTEND_API_CALL_MAP.md (Not counted above)
**Visual architecture** - Component and service relationships

Contains:
- Component → API call tree diagrams
- Service class flow diagrams
- Hook-based auto-refresh patterns
- Data pipeline workflows
- Error handling flow charts
- Cross-service communication diagram
- Request/response patterns
- Performance characteristics table

---

## At a Glance

| Metric | Value |
|--------|-------|
| Total API Endpoints | 45+ |
| GET Requests | 41 |
| POST Requests | 4 |
| WebSocket Connections | 1 |
| Service Categories | 8 |
| Backend Services | 3 (ports: 8000, 8001, 8082) |
| Documentation Lines | 996+ |
| Backend Ports Used | 8000, 8001, 8082, 9000, 5432 |

---

## API Categories Summary

| Category | Endpoints | File Reference |
|----------|-----------|-----------------|
| Market Data | 6 | API_CALLS_ANALYSIS.md §2.A |
| Pipeline (L0-L6) | 8 | API_CALLS_ANALYSIS.md §2.B |
| Trading Signals | 2 | API_CALLS_ANALYSIS.md §2.C |
| ML Analytics | 12 | API_CALLS_ANALYSIS.md §2.D |
| Backtest | 2 | API_CALLS_ANALYSIS.md §2.E |
| Health Checks | 6 | API_CALLS_ANALYSIS.md §2.F |
| Real-time Market | 4 | API_CALLS_ANALYSIS.md §2.G |
| Analytics Hooks | 6 | API_CALLS_ANALYSIS.md §2.H |
| **Total** | **46+** | |

---

## Key Endpoints

### Most Frequently Called
- `/api/proxy/trading/candlesticks/{symbol}` - OHLC data
- `/api/proxy/trading/latest/{symbol}` - Real-time price
- `/api/proxy/ws` - WebSocket proxy
- `/api/pipeline/l0/raw-data` - Raw data access

### Real-time Data
- `ws://48.216.199.139:8082/ws` - WebSocket (live updates)
- `/api/proxy/ws` - Polling fallback

### Analytics Data
- `/api/ml-analytics/predictions` - ML predictions
- `/api/ml-analytics/health` - Model health
- `{ANALYTICS_API}/api/analytics/session-pnl` - Session P&L

### Pipeline Layers
- L0: `/api/pipeline/l0/raw-data` - Raw market data
- L1: `/api/pipeline/l1/episodes` - RL episodes
- L3: `/api/pipeline/l3/features` - Feature correlation
- L4: `/api/pipeline/l4/dataset` - RL-ready dataset
- L5: `/api/pipeline/l5/models` - Model artifacts
- L6: `/api/pipeline/l6/backtest-results` - Backtest results

---

## Service Layer Architecture

```
Frontend (React)
    ↓
Services (lib/services/)
├─ MarketDataService - Market data with fallbacks
├─ EnhancedDataService - Historical data
├─ BacktestClient - Backtest operations
└─ PipelineDataClient - Pipeline access
    ↓
Next.js API Routes (/app/api/)
├─ Proxy routes (/proxy/*)
├─ Pipeline routes (/pipeline/*)
├─ Analytics routes (/ml-analytics/*)
└─ Backtest routes (/backtest/*)
    ↓
Backend Services
├─ Trading API (port 8000) → PostgreSQL
├─ Analytics API (port 8001) → ML Results
├─ WebSocket (port 8082) → Real-time
└─ MinIO (port 9000) → Storage
```

---

## HTTP Methods

### GET (41 endpoints)
- Market data queries
- Pipeline data retrieval
- Health checks
- ML analytics
- Status endpoints

### POST (4 endpoints)
- `/api/backtest/trigger` - Trigger backtest
- `/api/ml-analytics/health` - Report health
- `/api/ml-analytics/predictions` - Store predictions
- `/api/market/update` - Update market data

### WebSocket (1 connection)
- `ws://48.216.199.139:8082/ws` - Real-time price streaming

---

## Component Coverage

### Components Making API Calls
- 20+ Dashboard view components
- 6 ML analytics components
- 4 Trading components
- 2 Status components
- Multiple chart components

### Service Implementations
- 4 Main service classes
- 2 Custom React hooks (with auto-refresh)
- Multiple API route handlers

---

## Quick Access Guide

### I want to...

**Find an endpoint**
→ Use `API_QUICK_REFERENCE.md`

**Understand how data flows**
→ Use `FRONTEND_API_CALL_MAP.md`

**Get detailed technical info**
→ Use `API_CALLS_ANALYSIS.md`

**Debug an API issue**
→ Use `README_API_ANALYSIS.md` troubleshooting section

**See which component calls an API**
→ Use `FRONTEND_API_CALL_MAP.md` component sections

**Understand error handling**
→ Use `README_API_ANALYSIS.md` error handling chains

**Check refresh intervals**
→ Use `README_API_ANALYSIS.md` or `API_QUICK_REFERENCE.md`

**Test endpoints locally**
→ Use `README_API_ANALYSIS.md` testing section

---

## Environment Setup

Configuration file: `.env.local`

```
NEXT_PUBLIC_TRADING_API_URL=/api/proxy/trading
NEXT_PUBLIC_WS_URL=ws://48.216.199.139:8082
NEXT_PUBLIC_ANALYTICS_API_URL=http://localhost:8001
NEXT_PUBLIC_APP_NAME=USDCOP Trading Dashboard
NEXT_PUBLIC_DEFAULT_SYMBOL=USDCOP
NEXT_PUBLIC_DEFAULT_TIMEFRAME=5m
```

---

## Backend Services (Must be running)

| Service | Port | Purpose |
|---------|------|---------|
| Trading API | 8000 | Market data, OHLC, candlesticks |
| Analytics API | 8001 | ML metrics, risk, performance |
| WebSocket | 8082 | Real-time data streaming |
| MinIO | 9000 | Object storage for market data |
| PostgreSQL | 5432 | Timescale DB (92K+ market records) |

---

## Data Refresh Rates

| Data Type | Interval |
|-----------|----------|
| Real-time (WebSocket) | Live/tick |
| Session P&L | 30 seconds |
| Market Stats | 30 seconds |
| Risk Metrics | 60 seconds |
| Performance KPIs | 120 seconds |
| Production Gates | 120 seconds |
| Trading Signals | On demand |
| Pipeline Data | On demand |
| Health Checks | On demand |

---

## Common Query Parameters

- `limit` - Max records (1-10000)
- `offset` - Pagination offset
- `symbol` - Trading pair (default: USDCOP)
- `timeframe` - 5m, 15m, 1h, 4h, 1d
- `start_date` - ISO date format
- `end_date` - ISO date format
- `split` - train/test/val
- `source` - postgres/minio/twelvedata/all
- `action` - Specific endpoint action

---

## Authentication

Most endpoints are **open** (no authentication required).

Default headers:
```javascript
{
  'Content-Type': 'application/json'
}
```

---

## Error Handling Strategy

All services implement multi-level fallbacks:

1. Try primary API endpoint
2. Try secondary source or alternative endpoint
3. Try historical/cached data
4. Use mock/default data
5. Return error state

This ensures the dashboard remains functional even when some backend services are unavailable.

---

## Performance Characteristics

| Endpoint Type | Latency | Cache | Refresh |
|---|---|---|---|
| Real-time (WS) | <100ms | N/A | Live |
| Market Data | 100-500ms | 2-30s | 30s-120s |
| ML Analytics | 200-1000ms | 1min | 60s-120s |
| Pipeline Data | 500-2000ms | 5min-1hr | On demand |
| Health | 100-200ms | 1min | On demand |

---

## Testing & Debugging

### Check API Health
```bash
curl http://localhost:3000/api/market/health
curl http://localhost:3000/api/pipeline/l0/raw-data?limit=1
```

### Check WebSocket
Browser DevTools → Network → WS tab

### View API Calls
Browser DevTools → Network tab → filter by XHR/Fetch

### Check Error Logs
Browser Console → Network failures

---

## Document Maintenance

Last updated: October 21, 2025

These documents are automatically generated from:
- `/lib/services/*.ts` - Service implementations
- `/hooks/*.ts` - React hooks
- `/app/api/**/*.ts` - API routes
- `/components/**/*.tsx` - Component implementations

---

## Navigation

- **Start Here:** README_API_ANALYSIS.md
- **Quick Lookup:** API_QUICK_REFERENCE.md
- **Visual Architecture:** FRONTEND_API_CALL_MAP.md
- **Technical Details:** API_CALLS_ANALYSIS.md

