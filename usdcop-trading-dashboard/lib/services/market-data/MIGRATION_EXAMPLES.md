# Migration Examples - Market Data Services

This guide shows how to migrate from the monolithic `MarketDataService` to the new modular structure.

## Table of Contents
1. [No Changes Needed (Backwards Compatible)](#no-changes-needed)
2. [Migrating to Modular Imports](#migrating-to-modular-imports)
3. [Real-World Examples](#real-world-examples)
4. [Common Patterns](#common-patterns)

---

## No Changes Needed (Backwards Compatible)

The original API still works! No immediate changes required:

```typescript
// ✅ Still works - no changes needed
import { MarketDataService } from '@/lib/services/market-data-service'

const data = await MarketDataService.getRealTimeData('USDCOP')
const formatted = MarketDataService.formatPrice(4123.45)
```

---

## Migrating to Modular Imports

For better tree-shaking and clearer code organization:

### Before (Monolithic)
```typescript
import { MarketDataService } from '@/lib/services/market-data-service'

// All methods accessed through static class
const data = await MarketDataService.getRealTimeData('USDCOP')
const formatted = MarketDataService.formatPrice(data[0].price)
const stats = await MarketDataService.getSymbolStats('USDCOP')
const change = MarketDataService.calculatePriceChange(current, previous)
```

### After (Modular)
```typescript
import {
  MarketDataFetcher,
  DataTransformer,
  StatisticsCalculator
} from '@/lib/services/market-data'

// Create fetcher instance (reusable)
const fetcher = new MarketDataFetcher({
  apiBaseUrl: '/api/proxy/trading'
})

// Use specialized modules
const data = await fetcher.getRealTimeData('USDCOP')
const formatted = DataTransformer.formatPrice(data[0].price)
const stats = await fetcher.getSymbolStats('USDCOP')
const change = StatisticsCalculator.calculatePriceChange(current, previous)
```

---

## Real-World Examples

### Example 1: Real-Time Price Display Component

**Before:**
```typescript
import { MarketDataService } from '@/lib/services/market-data-service'
import { useEffect, useState } from 'react'

export function PriceDisplay() {
  const [price, setPrice] = useState<number>(0)

  useEffect(() => {
    const unsubscribe = MarketDataService.subscribeToRealTimeUpdates((data) => {
      setPrice(data.price)
    })
    return unsubscribe
  }, [])

  return <div>{MarketDataService.formatPrice(price)}</div>
}
```

**After (Recommended):**
```typescript
import { MarketDataFetcher, DataTransformer } from '@/lib/services/market-data'
import { useEffect, useState, useMemo } from 'react'

export function PriceDisplay() {
  const [price, setPrice] = useState<number>(0)

  // Create fetcher once
  const fetcher = useMemo(
    () => new MarketDataFetcher({ apiBaseUrl: '/api/proxy/trading' }),
    []
  )

  useEffect(() => {
    const unsubscribe = fetcher.subscribeToPolling((data) => {
      setPrice(data.price)
    }, 2000)
    return unsubscribe
  }, [fetcher])

  return <div>{DataTransformer.formatPrice(price)}</div>
}
```

**Benefits:**
- Only imports what you need (smaller bundle)
- Fetcher instance is reusable
- Clear separation of concerns

---

### Example 2: Chart Component with Statistics

**Before:**
```typescript
import { MarketDataService } from '@/lib/services/market-data-service'

export async function loadChartData(symbol: string) {
  const candles = await MarketDataService.getCandlestickData(symbol, '5m', undefined, undefined, 100)
  const stats = await MarketDataService.getSymbolStats(symbol)

  // Calculate additional stats
  const prices = candles.data.map(c => c.close)
  const change = MarketDataService.calculatePriceChange(
    prices[0],
    prices[prices.length - 1]
  )

  return { candles, stats, change }
}
```

**After (Recommended):**
```typescript
import {
  MarketDataFetcher,
  StatisticsCalculator,
  type CandlestickResponse,
  type SymbolStats
} from '@/lib/services/market-data'

const fetcher = new MarketDataFetcher({ apiBaseUrl: '/api/proxy/trading' })

export async function loadChartData(symbol: string) {
  const candles = await fetcher.getCandlestickData(symbol, '5m', undefined, undefined, 100)
  const stats = await fetcher.getSymbolStats(symbol)

  // Calculate additional stats
  const prices = candles.data.map(c => c.close)
  const change = StatisticsCalculator.calculatePriceChange(
    prices[0],
    prices[prices.length - 1]
  )

  return { candles, stats, change }
}
```

**Benefits:**
- Better TypeScript intellisense
- Explicit module imports
- Easier to mock for testing

---

### Example 3: WebSocket Connection

**Before:**
```typescript
import { MarketDataService } from '@/lib/services/market-data-service'

export function connectToMarket() {
  const ws = MarketDataService.connectWebSocket()

  if (!ws) {
    console.error('Failed to connect')
    return
  }

  // No direct control over reconnection or connection state
}
```

**After (Recommended):**
```typescript
import { WebSocketConnector } from '@/lib/services/market-data'

export function connectToMarket() {
  const connector = new WebSocketConnector({
    url: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8082',
    reconnectInterval: 5000,
    autoReconnect: true
  })

  // Full control over connection
  connector.connect('USDCOP')

  // Subscribe to updates
  connector.addSubscriber((data) => {
    console.log('Received:', data)
  })

  // Check connection state
  if (connector.isConnected()) {
    console.log('Connected successfully')
  }

  // Disconnect when needed
  return () => connector.disconnect()
}
```

**Benefits:**
- Full control over WebSocket lifecycle
- Better error handling
- Can create multiple connections if needed

---

### Example 4: Statistics Dashboard

**Before:**
```typescript
import { MarketDataService } from '@/lib/services/market-data-service'

export async function DashboardStats({ symbol }: { symbol: string }) {
  const candles = await MarketDataService.getCandlestickData(symbol, '5m', undefined, undefined, 288)
  const prices = candles.data.map(c => c.close)

  // All calculations through one service
  const current = prices[prices.length - 1]
  const previous = prices[0]
  const change = MarketDataService.calculatePriceChange(current, previous)

  return (
    <div>
      <div>Price: {MarketDataService.formatPrice(current)}</div>
      <div>Change: {change.changePercent.toFixed(2)}%</div>
    </div>
  )
}
```

**After (Recommended):**
```typescript
import {
  MarketDataFetcher,
  StatisticsCalculator,
  DataTransformer
} from '@/lib/services/market-data'

const fetcher = new MarketDataFetcher({ apiBaseUrl: '/api/proxy/trading' })

export async function DashboardStats({ symbol }: { symbol: string }) {
  const candles = await fetcher.getCandlestickData(symbol, '5m', undefined, undefined, 288)

  // Use specialized calculators
  const stats24h = StatisticsCalculator.calculate24hStats(candles.data)
  const priceStats = StatisticsCalculator.calculatePriceStats(candles.data)
  const volumeStats = StatisticsCalculator.calculateVolumeStats(candles.data)

  return (
    <div>
      <div>Price: {DataTransformer.formatPrice(stats24h.close)}</div>
      <div>Change: {DataTransformer.formatPercent(stats24h.changePercent)}</div>
      <div>Volume: {DataTransformer.formatVolume(volumeStats.total)}</div>
      <div>Volatility: {priceStats.stdDev.toFixed(4)}</div>
    </div>
  )
}
```

**Benefits:**
- More statistics available
- Clearer code organization
- Easier to add new metrics

---

## Common Patterns

### Pattern 1: Custom Hook for Market Data

```typescript
import { useEffect, useState, useMemo } from 'react'
import { MarketDataFetcher, type MarketDataPoint } from '@/lib/services/market-data'

export function useMarketData(symbol: string, pollInterval: number = 2000) {
  const [data, setData] = useState<MarketDataPoint | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)

  const fetcher = useMemo(
    () => new MarketDataFetcher({ apiBaseUrl: '/api/proxy/trading' }),
    []
  )

  useEffect(() => {
    let isSubscribed = true

    // Initial fetch
    fetcher.getRealTimeData(symbol)
      .then(data => {
        if (isSubscribed) {
          setData(data[0])
          setLoading(false)
        }
      })
      .catch(err => {
        if (isSubscribed) {
          setError(err)
          setLoading(false)
        }
      })

    // Subscribe to updates
    const unsubscribe = fetcher.subscribeToPolling((newData) => {
      if (isSubscribed) {
        setData(newData)
      }
    }, pollInterval)

    return () => {
      isSubscribed = false
      unsubscribe()
    }
  }, [symbol, pollInterval, fetcher])

  return { data, loading, error }
}
```

### Pattern 2: Service Container (Dependency Injection)

```typescript
import {
  MarketDataFetcher,
  WebSocketConnector,
  type FetcherConfig,
  type WebSocketConfig
} from '@/lib/services/market-data'

export class MarketDataContainer {
  private fetcher: MarketDataFetcher
  private wsConnector: WebSocketConnector | null = null

  constructor(
    fetcherConfig: FetcherConfig,
    wsConfig?: WebSocketConfig
  ) {
    this.fetcher = new MarketDataFetcher(fetcherConfig)

    if (wsConfig && typeof window !== 'undefined') {
      this.wsConnector = new WebSocketConnector(wsConfig)
    }
  }

  getFetcher() {
    return this.fetcher
  }

  getWSConnector() {
    return this.wsConnector
  }

  async initialize(symbol: string) {
    // Check API health
    const health = await this.fetcher.checkAPIHealth()
    console.log('API Health:', health)

    // Connect WebSocket if available
    if (this.wsConnector) {
      this.wsConnector.connect(symbol)
    }
  }

  cleanup() {
    this.wsConnector?.disconnect()
  }
}

// Usage:
const container = new MarketDataContainer(
  { apiBaseUrl: '/api/proxy/trading' },
  { url: 'ws://localhost:8082', autoReconnect: true }
)
```

### Pattern 3: Caching Layer

```typescript
import { MarketDataFetcher, type CandlestickResponse } from '@/lib/services/market-data'

class CachedMarketDataFetcher extends MarketDataFetcher {
  private cache = new Map<string, { data: CandlestickResponse; timestamp: number }>()
  private cacheTTL = 60000 // 1 minute

  async getCandlestickData(
    symbol: string,
    timeframe: string = '5m',
    startDate?: string,
    endDate?: string,
    limit: number = 1000,
    includeIndicators: boolean = true
  ): Promise<CandlestickResponse> {
    const cacheKey = `${symbol}-${timeframe}-${limit}`
    const cached = this.cache.get(cacheKey)

    if (cached && Date.now() - cached.timestamp < this.cacheTTL) {
      return cached.data
    }

    const data = await super.getCandlestickData(
      symbol,
      timeframe,
      startDate,
      endDate,
      limit,
      includeIndicators
    )

    this.cache.set(cacheKey, { data, timestamp: Date.now() })
    return data
  }
}
```

---

## Summary

### When to Use Each Approach:

1. **Backwards Compatible (Original API)**
   - Quick fixes
   - Legacy code
   - No time for refactoring

2. **Modular Imports (Recommended)**
   - New features
   - Performance-critical code
   - Testing required
   - Clear separation of concerns

3. **Custom Patterns**
   - Complex applications
   - Multiple symbols/connections
   - Advanced caching needs
   - Dependency injection

### Migration Checklist:

- [ ] Identify all imports of `MarketDataService`
- [ ] Determine which methods are used
- [ ] Import only the required modules
- [ ] Update method calls to use module instances
- [ ] Add proper TypeScript types
- [ ] Test thoroughly
- [ ] Update documentation

### Questions?

Check the main [README.md](./README.md) for complete API documentation.
