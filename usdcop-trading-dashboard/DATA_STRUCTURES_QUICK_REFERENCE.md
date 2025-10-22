# DATA STRUCTURES QUICK REFERENCE
## Interfaces y Tipos - Referencia Rapida

---

## TABLA 1: INTERFACES POR ARCHIVO

| Archivo | Interface | Tipo | Defaults | Mock Data | API Endpoint |
|---------|-----------|------|----------|-----------|--------------|
| **useMarketStats.ts** | MarketStats | Hook State | ✅ Si | ❌ No | GET /api/stats/{symbol} (8000) |
| **useAnalytics.ts** | RLMetrics | API Response | ❌ No | ❌ No | GET /rl-metrics (8001) |
| **useAnalytics.ts** | PerformanceKPIs | API Response | ❌ No | ❌ No | GET /performance-kpis (8001) |
| **useAnalytics.ts** | ProductionGate | API Response | ✅ Partial | ❌ No | GET /production-gates (8001) |
| **useAnalytics.ts** | RiskMetrics | API Response | ❌ No | ❌ No | GET /risk-metrics (8001) |
| **useAnalytics.ts** | SessionPnL | API Response | ✅ Partial | ❌ No | GET /session-pnl (8001) |
| **useRealtimeData.ts** | MarketData | WebSocket Data | ✅ Si | ❌ No | WS ws://localhost:3001 |
| **useRealTimePrice.ts** | RealTimePriceState | Hook State | ✅ Si | ❌ No | MarketDataService |
| **real-time-risk-engine.ts** | Position | Data Structure | N/A | ❌ No | N/A |
| **real-time-risk-engine.ts** | RiskAlert | Generated | N/A | ❌ No | N/A |
| **real-time-risk-engine.ts** | RealTimeRiskMetrics | API Response | ✅ Si | ❌ No | GET /risk-metrics (8001) |
| **real-market-metrics.ts** | OHLCData | Input Data | N/A | ❌ No | N/A |
| **real-market-metrics.ts** | RealMarketMetrics | Calculated | N/A | ❌ No | Calculated from OHLC |
| **market-data-service.ts** | MarketDataPoint | API Response | ✅ Si | ❌ No | GET /latest/{symbol} (8000) |
| **market-data-service.ts** | CandlestickData | API Response | N/A | ❌ No | GET /candlesticks/{symbol} (8000) |
| **market-data-service.ts** | TechnicalIndicators | API Response | N/A | ❌ No | Included in candlesticks |
| **backtest-client.ts** | BacktestResult | API Response | N/A | ✅ Si | GET /api/backtest/results |
| **backtest-client.ts** | BacktestResults | API Response | N/A | ✅ Si | GET /api/backtest/results |
| **backtest-client.ts** | TradeRecord | Data Structure | N/A | ✅ Si | Part of BacktestResults |
| **backtest-client.ts** | DailyReturn | Data Structure | N/A | ✅ Si | Part of BacktestResults |
| **backtest-client.ts** | BacktestKPIs | Data Structure | N/A | ✅ Si | Part of BacktestResults |
| **hedge-fund-metrics.ts** | HedgeFundMetrics | Calculated | N/A | ❌ No | Calculated from trades |
| **hedge-fund-metrics.ts** | Trade | Data Structure | N/A | ❌ No | N/A |
| **pipeline-data-client.ts** | PipelineDataPoint | API Response | N/A | ✅ Si | Should be /api/pipeline/data (8002) |
| **enhanced-data-service.ts** | MarketData | API Response | N/A | ⚠️ Fallback | GET /api/market/historical (8000) |
| **enhanced-data-service.ts** | EnhancedCandle | API Response | N/A | ⚠️ Fallback | GET /api/market/complete-history (8000) |
| **libs/core/types/market-data.ts** | MarketTick | Type Definition | N/A | ❌ No | N/A |
| **libs/core/types/market-data.ts** | OHLCV | Type Definition | N/A | ❌ No | N/A |
| **libs/core/types/market-data.ts** | OrderBook | Type Definition | N/A | ❌ No | N/A |
| **libs/core/types/market-data.ts** | Trade | Type Definition | N/A | ❌ No | N/A |
| **libs/core/types/market-data.ts** | MarketDepth | Type Definition | N/A | ❌ No | N/A |
| **libs/core/types/market-data.ts** | MarketSession | Type Definition | N/A | ❌ No | N/A |
| **libs/core/types/market-data.ts** | MarketStats | Type Definition | N/A | ❌ No | N/A |
| **libs/core/types/market-data.ts** | StreamMetadata | Type Definition | N/A | ❌ No | N/A |
| **libs/core/types/index.ts** | ApiResponse\<T\> | Type Definition | N/A | ❌ No | N/A |
| **libs/core/types/index.ts** | PaginatedResponse\<T\> | Type Definition | N/A | ❌ No | N/A |
| **libs/core/types/index.ts** | CacheEntry\<T\> | Type Definition | N/A | ❌ No | N/A |
| **libs/core/types/index.ts** | WebSocketMessage\<T\> | Type Definition | N/A | ❌ No | N/A |

**Leyenda:**
- ✅ Si = Tiene defaults correctos
- ❌ No = No tiene defaults (o no aplica)
- ⚠️ Fallback = Tiene fallback a mock data
- N/A = No aplica (type definitions, calculated values)

---

## TABLA 2: PROBLEMAS POR PRIORIDAD

### PRIORIDAD ALTA 🔴

| # | Archivo | Problema | Solucion |
|---|---------|----------|----------|
| 1 | backtest-client.ts | getLatestResults() genera mock KPIs con valores hardcoded (CAGR: 0.125, Sharpe: 1.45, etc.) | Retornar null si API falla, no mock data |
| 2 | pipeline-data-client.ts | getPipelineData() siempre retorna los mismos 2 puntos hardcoded | Conectar con Pipeline API (puerto 8002) |
| 3 | useAnalytics.ts | RLMetrics, PerformanceKPIs, RiskMetrics sin defaults | Agregar DEFAULT_* constants |

### PRIORIDAD MEDIA 🟡

| # | Archivo | Problema | Solucion |
|---|---------|----------|----------|
| 4 | enhanced-data-service.ts | getHistoricalData() genera 100 puntos aleatorios como fallback | Eliminar fallback, throw error si API falla |
| 5 | useAnalytics.ts | Hook retorna undefined hasta que carga, puede causar undefined errors en UI | Usar defaults mientras carga |

### PRIORIDAD BAJA 🟢

| # | Archivo | Problema | Solucion |
|---|---------|----------|----------|
| 6 | Todas las interfaces | Falta documentacion JSDoc | Agregar comments con descripcion y unidades |
| 7 | API responses | No hay validacion runtime | Implementar Zod schemas |

---

## TABLA 3: INTERFACES CON DEFAULTS CORRECTOS ✅

| Interface | Archivo | Defaults |
|-----------|---------|----------|
| MarketStats | useMarketStats.ts | Todos en 0, trend: 'neutral', source: 'initializing' |
| RealTimePriceState | useRealTimePrice.ts | null para datos, false para flags |
| MarketData (array) | useRealtimeData.ts | [] (array vacio) |
| ProductionGate (array) | useAnalytics.ts | [] (array vacio) |
| SessionPnL (partial) | useAnalytics.ts | pnl: 0, pnlPercent: 0, hasData: false |
| RealTimeRiskMetrics | real-time-risk-engine.ts | Fetched de API, null si falla |
| MarketDataPoint | market-data-service.ts | Usa ultimo dato real de DB si market closed |

---

## TABLA 4: INTERFACES SIN DEFAULTS (NECESITAN CORRECCION) ❌

| Interface | Archivo | Default Actual | Default Recomendado |
|-----------|---------|----------------|---------------------|
| RLMetrics | useAnalytics.ts | undefined | { tradesPerEpisode: 0, avgHolding: 0, actionBalance: {buy:0,sell:0,hold:0}, spreadCaptured: 0, pegRate: 0, vwapError: 0 } |
| PerformanceKPIs | useAnalytics.ts | undefined | { sortinoRatio: 0, calmarRatio: 0, sharpeRatio: 0, maxDrawdown: 0, currentDrawdown: 0, profitFactor: 0, cagr: 0, volatility: 0, benchmarkSpread: 0 } |
| RiskMetrics | useAnalytics.ts | undefined | { portfolioValue: 0, grossExposure: 0, netExposure: 0, leverage: 0, portfolioVaR95: 0, portfolioVaR99: 0, portfolioVaR95Percent: 0, expectedShortfall95: 0, portfolioVolatility: 0, currentDrawdown: 0, maximumDrawdown: 0, liquidityScore: 0, timeToLiquidate: 0, bestCaseScenario: 0, worstCaseScenario: 0, stressTestResults: {} } |

---

## TABLA 5: CODIGO PARA COPIAR/PEGAR

### Agregar a useAnalytics.ts

```typescript
// ============================================
// DEFAULT VALUES FOR ANALYTICS INTERFACES
// ============================================

const DEFAULT_RL_METRICS: RLMetrics = {
  tradesPerEpisode: 0,
  avgHolding: 0,
  actionBalance: {
    buy: 0,
    sell: 0,
    hold: 0
  },
  spreadCaptured: 0,
  pegRate: 0,
  vwapError: 0
}

const DEFAULT_PERFORMANCE_KPIS: PerformanceKPIs = {
  sortinoRatio: 0,
  calmarRatio: 0,
  sharpeRatio: 0,
  maxDrawdown: 0,
  currentDrawdown: 0,
  profitFactor: 0,
  cagr: 0,
  volatility: 0,
  benchmarkSpread: 0
}

const DEFAULT_RISK_METRICS: RiskMetrics = {
  portfolioValue: 0,
  grossExposure: 0,
  netExposure: 0,
  leverage: 0,
  portfolioVaR95: 0,
  portfolioVaR99: 0,
  portfolioVaR95Percent: 0,
  expectedShortfall95: 0,
  portfolioVolatility: 0,
  currentDrawdown: 0,
  maximumDrawdown: 0,
  liquidityScore: 0,
  timeToLiquidate: 0,
  bestCaseScenario: 0,
  worstCaseScenario: 0,
  stressTestResults: {}
}

// ============================================
// UPDATED HOOKS WITH DEFAULTS
// ============================================

export function useRLMetrics(symbol: string = 'USDCOP', days: number = 30) {
  const { data, error, isLoading } = useSWR<RLMetricsResponse>(
    `${ANALYTICS_API_URL}/rl-metrics?symbol=${symbol}&days=${days}`,
    fetcher,
    {
      refreshInterval: 60000,
      revalidateOnFocus: false,
    }
  );

  return {
    metrics: data?.metrics || DEFAULT_RL_METRICS,  // ✅ Added default
    isLoading,
    isError: error,
    raw: data,
  };
}

export function usePerformanceKPIs(symbol: string = 'USDCOP', days: number = 90) {
  const { data, error, isLoading } = useSWR<PerformanceKPIsResponse>(
    `${ANALYTICS_API_URL}/performance-kpis?symbol=${symbol}&days=${days}`,
    fetcher,
    {
      refreshInterval: 120000,
      revalidateOnFocus: false,
    }
  );

  return {
    kpis: data?.kpis || DEFAULT_PERFORMANCE_KPIS,  // ✅ Added default
    isLoading,
    isError: error,
    raw: data,
  };
}

export function useRiskMetrics(
  symbol: string = 'USDCOP',
  portfolioValue: number = 10000000,
  days: number = 30
) {
  const { data, error, isLoading } = useSWR<RiskMetricsResponse>(
    `${ANALYTICS_API_URL}/risk-metrics?symbol=${symbol}&portfolio_value=${portfolioValue}&days=${days}`,
    fetcher,
    {
      refreshInterval: 60000,
      revalidateOnFocus: false,
    }
  );

  return {
    metrics: data?.risk_metrics || DEFAULT_RISK_METRICS,  // ✅ Added default
    isLoading,
    isError: error,
    raw: data,
  };
}
```

### Reemplazar en backtest-client.ts

```typescript
export const backtestClient = {
  async getLatestResults(): Promise<BacktestResults | null> {
    try {
      const response = await fetch('/api/backtest/results')

      if (!response.ok) {
        console.error(`[BacktestClient] HTTP ${response.status}`)
        return null  // ✅ Return null instead of mock
      }

      const apiResult = await response.json()

      if (apiResult.success && apiResult.data) {
        console.log('[BacktestClient] Successfully fetched results from API')
        return apiResult.data
      }

      return null  // ✅ Return null if no data
    } catch (error) {
      console.error('[BacktestClient] API error:', error)
      return null  // ✅ Return null on error
    }
  }
}

// Usage in components:
const results = await backtestClient.getLatestResults()
if (!results) {
  return <div>No backtest data available. Run a backtest first.</div>
}
```

### Reemplazar en pipeline-data-client.ts

```typescript
export async function getPipelineData(): Promise<PipelineDataPoint[]> {
  try {
    const PIPELINE_API_URL = process.env.NEXT_PUBLIC_PIPELINE_API_URL || 'http://localhost:8002'
    const response = await fetch(`${PIPELINE_API_URL}/api/pipeline/data`)

    if (!response.ok) {
      console.error(`[PipelineData] HTTP ${response.status}`)
      return []  // ✅ Return empty array
    }

    const data = await response.json()
    console.log('[PipelineData] Successfully fetched from API')
    return data
  } catch (error) {
    console.error('[PipelineData] Failed to fetch:', error)
    return []  // ✅ Return empty array on error
  }
}

export const pipelineDataService = {
  getPipelineData,

  async getHealthStatus() {
    try {
      const PIPELINE_API_URL = process.env.NEXT_PUBLIC_PIPELINE_API_URL || 'http://localhost:8002'
      const response = await fetch(`${PIPELINE_API_URL}/health`)
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      return await response.json()
    } catch (error) {
      console.error('[PipelineData] Health check failed:', error)
      return { status: 'error', timestamp: Date.now() }
    }
  },

  async getMetrics() {
    try {
      const PIPELINE_API_URL = process.env.NEXT_PUBLIC_PIPELINE_API_URL || 'http://localhost:8002'
      const response = await fetch(`${PIPELINE_API_URL}/metrics`)
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      return await response.json()
    } catch (error) {
      console.error('[PipelineData] Metrics fetch failed:', error)
      return { processed: 0, errors: 0, latency: 0 }
    }
  }
}
```

### Reemplazar en enhanced-data-service.ts

```typescript
async getHistoricalData(
  symbol: string,
  startTime: number,
  endTime: number
): Promise<MarketData[]> {
  const response = await fetch(
    `/api/proxy/trading/api/market/historical?symbol=${symbol}&start=${startTime}&end=${endTime}`
  )

  if (!response.ok) {
    throw new Error(`Failed to fetch historical data: HTTP ${response.status}`)
  }

  const data = await response.json()
  console.log(`[EnhancedDataService] Fetched ${data.length} historical points`)
  return data
}

async loadCompleteHistory(symbol: string = 'USDCOP'): Promise<EnhancedCandle[]> {
  const response = await fetch(
    `/api/proxy/trading/api/market/complete-history?symbol=${symbol}`
  )

  if (!response.ok) {
    throw new Error(`Failed to fetch complete history: HTTP ${response.status}`)
  }

  const data = await response.json()
  console.log(`[EnhancedDataService] Loaded ${data.length} candles`)
  return data
}
```

---

## TABLA 6: MAPEO INTERFACE -> API

| Interface | Backend Service | Port | Endpoint | Method |
|-----------|----------------|------|----------|--------|
| MarketStats | Trading API | 8000 | /api/stats/{symbol} | GET |
| MarketDataPoint | Trading API | 8000 | /api/latest/{symbol} | GET |
| CandlestickData | Trading API | 8000 | /api/candlesticks/{symbol} | GET |
| RLMetrics | Analytics API | 8001 | /rl-metrics | GET |
| PerformanceKPIs | Analytics API | 8001 | /performance-kpis | GET |
| ProductionGate | Analytics API | 8001 | /production-gates | GET |
| RiskMetrics | Analytics API | 8001 | /risk-metrics | GET |
| SessionPnL | Analytics API | 8001 | /session-pnl | GET |
| BacktestResults | Backtest API | TBD | /api/backtest/results | GET |
| PipelineDataPoint | Pipeline API | 8002 | /api/pipeline/data | GET |
| MarketData (realtime) | WebSocket | 3001 | /ws | WS |

---

## TABLA 7: TIPOS DE VALORES POR DEFECTO

| Tipo de Campo | Default Recomendado | Ejemplos |
|---------------|---------------------|----------|
| number (price, value) | 0 | currentPrice: 0, portfolioValue: 0 |
| number (percentage) | 0 | changePercent: 0, volatility: 0 |
| number (ratio) | 0 | sharpeRatio: 0, profitFactor: 0 |
| string | '' o 'initializing' | source: 'initializing' |
| boolean | false | isConnected: false, acknowledged: false |
| Date | new Date() | timestamp: new Date() |
| Array | [] | gates: [], trades: [] |
| Object | {} | actionBalance: {buy:0,sell:0,hold:0} |
| Optional data | null | currentPrice: null, error: null |
| Union type | 'neutral' | trend: 'neutral' |

---

## CHECKLIST DE VALIDACION

Usar este checklist cuando crees una nueva interface:

### Interface Definition
- [ ] Nombre descriptivo (PascalCase)
- [ ] Exportada con `export interface`
- [ ] Todos los campos tipados explicitamente
- [ ] Campos opcionales marcados con `?`
- [ ] JSDoc comment con descripcion

### Default Values
- [ ] Numeros default a 0
- [ ] Strings default a '' o valor semantico
- [ ] Booleans default a false
- [ ] Arrays default a []
- [ ] Objects default a {} o estructura vacia
- [ ] Datos opcionales default a null
- [ ] NO usar valores mock/aleatorios

### API Integration
- [ ] Interface coincide con response de API
- [ ] Campos transformados correctamente (snake_case -> camelCase)
- [ ] Manejo de errores sin fallback a mock
- [ ] Tipos Date vs number para timestamps consistentes
- [ ] URL de API desde variable de entorno

### Code Quality
- [ ] Considera usar `readonly` para immutability
- [ ] Generic types si aplica (`ApiResponse<T>`)
- [ ] Union types para enums limitados
- [ ] No duplicar interfaces similares
- [ ] Reutilizar tipos de libs/core/types cuando aplique

---

**Version:** 1.0
**Ultima Actualizacion:** 2025-10-21
**Mantenido por:** AGENTE 4
