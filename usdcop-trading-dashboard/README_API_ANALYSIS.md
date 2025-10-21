# Frontend API Analysis - Complete Documentation

This directory contains comprehensive documentation of all API calls made from the USDCOP Trading Dashboard frontend.

## Documentation Files

### 1. **API_CALLS_ANALYSIS.md** - COMPREHENSIVE REFERENCE
The main analysis document containing:
- Complete API endpoint catalog with descriptions
- Query parameters and response formats
- Service-to-service communication patterns
- Component-level usage examples
- Error handling strategies
- Authentication details
- Response code meanings
- Performance characteristics

**Use this when you need:** Detailed information about a specific endpoint, understanding data flows, or comprehensive API reference.

### 2. **API_QUICK_REFERENCE.md** - QUICK LOOKUP
Fast reference guide with:
- All 45+ endpoints listed by category
- HTTP methods used
- Common query parameters
- Request/response body examples
- Backend services and ports
- Key files locations
- Refresh intervals

**Use this when you need:** Quick lookup of an endpoint, understanding the overall structure, or finding which component makes which call.

### 3. **FRONTEND_API_CALL_MAP.md** - VISUAL RELATIONSHIPS
Visual mapping showing:
- Component → API relationships (tree view)
- Service class → API flow diagrams
- Hook-based auto-refresh patterns
- Data pipeline workflows
- Error handling flow
- Cross-service communication diagram
- Request/response patterns
- Performance characteristics table

**Use this when you need:** Understanding which components call which endpoints, tracing data flows, or seeing the big picture architecture.

---

## Quick Facts

- **Total Endpoints:** 45+
- **HTTP Methods:** GET (41), POST (4), WebSocket (1)
- **Main Categories:** Market Data, Pipeline (L0-L6), Trading, ML Analytics, Backtest, Health
- **Backend Services:** 3 main (Trading API:8000, Analytics API:8001, WebSocket:8082)
- **Default Symbol:** USDCOP
- **Error Handling:** 100% with fallbacks

---

## API Categories at a Glance

| Category | Count | Endpoints | Purpose |
|----------|-------|-----------|---------|
| Market Data | 6 | `/api/proxy/trading/*` | Real-time prices, OHLC, stats |
| Pipeline (L0-L6) | 8 | `/api/pipeline/*` | Data processing layers |
| Trading Signals | 2 | `/api/trading/*` | Trading recommendations |
| ML Analytics | 12 | `/api/ml-analytics/*` | Model health, predictions, metrics |
| Backtest | 2 | `/api/backtest/*` | Backtesting operations |
| Health Checks | 6 | `/api/**/health`, `/api/status` | System status |
| Real-time Market | 4 | `/api/market/realtime`, `/api/market/update` | Live data streaming |
| External Analytics | 6 | `{ANALYTICS_API}/api/analytics/*` | RL metrics, risk, KPIs |

---

## Key Service Files

### Backend Service Implementations
- **lib/services/market-data-service.ts** - Market data, candlesticks, stats
- **lib/services/enhanced-data-service.ts** - Enhanced data with fallbacks
- **lib/services/backtest-client.ts** - Backtest trigger and results
- **lib/services/pipeline-data-client.ts** - Pipeline data access

### Frontend Hooks
- **hooks/useAnalytics.ts** - Analytics metrics (auto-refresh via SWR)
- **hooks/useMarketStats.ts** - Market statistics (auto-refresh via SWR)

### API Route Handlers
- **app/api/** - All Next.js API routes and proxy endpoints

### Components Making Calls
- **components/ml-analytics/** - Model performance, health, predictions
- **components/views/** - Pipeline dashboards (L0-L6)
- **components/trading/** - Trading signals, terminal
- **components/status/** - Connection status

---

## Common Usage Patterns

### Pattern 1: Fetch and Render
```typescript
const [data, setData] = useState(null);

useEffect(() => {
  fetch('/api/endpoint?param=value')
    .then(r => r.json())
    .then(data => setData(data))
    .catch(err => console.error(err));
}, []);

return <div>{data?.property}</div>;
```

### Pattern 2: SWR Hook (Auto-refresh)
```typescript
const { data, error, isLoading } = useSWR(
  '/api/endpoint?param=value',
  fetcher,
  { refreshInterval: 30000 } // Auto-refresh every 30s
);
```

### Pattern 3: Service Method with Fallbacks
```typescript
const data = await MarketDataService.getSymbolStats(symbol);
// Automatically tries multiple endpoints with fallbacks
```

### Pattern 4: WebSocket Real-time
```typescript
MarketDataService.connectWebSocket(); // ws://host:8082/ws
// Falls back to polling if WebSocket unavailable
```

---

## Environment Configuration

**File:** `.env.local`

```
NEXT_PUBLIC_TRADING_API_URL=/api/proxy/trading
NEXT_PUBLIC_WS_URL=ws://48.216.199.139:8082
NEXT_PUBLIC_APP_NAME=USDCOP Trading Dashboard
NEXT_PUBLIC_DEFAULT_SYMBOL=USDCOP
NEXT_PUBLIC_DEFAULT_TIMEFRAME=5m
NEXT_PUBLIC_ANALYTICS_API_URL=http://localhost:8001
```

---

## Data Flow Example: Loading Market Chart

```
User opens Chart Component
    ↓
Component: useMarketStats hook
    ↓
Service: MarketDataService.getCandlestickData(
  symbol='USDCOP',
  timeframe='5m',
  limit=1000
)
    ↓
API Call: GET /api/proxy/trading/candlesticks/USDCOP?
  timeframe=5m&limit=1000&include_indicators=true
    ↓
Backend Processing:
  1. PostgreSQL: SELECT FROM market_data
  2. Calculate Technical Indicators
  3. Format as OHLC
    ↓
Response:
{
  "symbol": "USDCOP",
  "timeframe": "5m",
  "data": [
    { time, open, high, low, close, volume, indicators: {...} }
  ]
}
    ↓
Component: Render candlestick chart with indicators
```

---

## Refresh Intervals (Data Update Rates)

| Data Type | Interval | Source |
|-----------|----------|--------|
| Real-time Price (WebSocket) | Live/Tick | Direct WS |
| Session P&L | 30 seconds | Analytics API |
| Market Stats | 30 seconds | Trading API |
| Risk Metrics | 60 seconds | Analytics API |
| Performance KPIs | 120 seconds | Analytics API |
| Production Gates | 120 seconds | Analytics API |
| Trading Signals | On demand | Trading API |
| Pipeline Data | On demand | Trading API |
| Health Checks | On demand | Health endpoints |

---

## Error Handling Chains

### Market Data Fallback Chain
1. WebSocket real-time connection
2. REST API real-time endpoint
3. Latest candlestick
4. Cached data
5. Mock/default data

### Pipeline Data Fallback Chain
1. PostgreSQL (primary)
2. MinIO (archive)
3. TwelveData API (external)
4. Empty/mock response

### Analytics Fallback Chain
1. Analytics API
2. Cached data
3. Default values

---

## Performance Tips

1. **Use SWR Hooks** - Automatic caching and revalidation
2. **Limit Query Results** - Use `limit` parameter to reduce payload
3. **Date Filters** - Use `start_date`/`end_date` to narrow results
4. **WebSocket Over Polling** - Use WS when available for lower latency
5. **Batch Requests** - Load multiple data types together where possible

---

## Testing Endpoints

### Local Development
```bash
# Trading API (port 8000)
curl http://localhost:8000/api/latest/USDCOP
curl http://localhost:8000/api/candlesticks/USDCOP?timeframe=5m&limit=10

# Analytics API (port 8001)
curl http://localhost:8001/api/analytics/session-pnl?symbol=USDCOP

# Frontend API (through Next.js proxy)
curl http://localhost:3000/api/pipeline/l0/raw-data?limit=10
```

---

## Troubleshooting

### API Endpoint Not Responding
1. Check backend service is running (ports 8000, 8001, 8082)
2. Verify environment variables are set correctly
3. Check network connectivity
4. Look at browser console for error messages

### Data Not Updating
1. Check refresh interval matches expected behavior
2. Verify WebSocket connection (open DevTools → Network → WS)
3. Check for errors in fallback chains
4. Verify data sources (PostgreSQL, MinIO) are accessible

### Performance Issues
1. Check number of records being fetched (use `limit`)
2. Review API response times in DevTools
3. Check if multiple components are making same request (use SWR)
4. Verify network bandwidth utilization

---

## Further Reading

- See `API_CALLS_ANALYSIS.md` for complete technical reference
- See `API_QUICK_REFERENCE.md` for quick lookups
- See `FRONTEND_API_CALL_MAP.md` for visual relationships

