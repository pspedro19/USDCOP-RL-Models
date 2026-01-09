# Market Data Services - Quick Reference

## Import Options

### Option 1: Backwards Compatible (No Changes)
```typescript
import { MarketDataService } from '@/lib/services/market-data-service'

// All original methods work
await MarketDataService.getRealTimeData('USDCOP')
MarketDataService.formatPrice(4123.45)
```

### Option 2: Modular Imports (Recommended)
```typescript
import {
  MarketDataFetcher,
  DataTransformer,
  StatisticsCalculator,
  WebSocketConnector
} from '@/lib/services/market-data'
```

### Option 3: Direct Imports (Best Tree-Shaking)
```typescript
import { DataTransformer } from '@/lib/services/market-data/DataTransformer'
import { StatisticsCalculator } from '@/lib/services/market-data/StatisticsCalculator'
```

---

## Module Cheat Sheet

### WebSocketConnector
```typescript
const ws = new WebSocketConnector({
  url: 'ws://localhost:8082',
  reconnectInterval: 5000,
  autoReconnect: true
})

ws.connect('USDCOP')
ws.addSubscriber((data) => console.log(data))
ws.isConnected() // true/false
ws.disconnect()
```

### MarketDataFetcher
```typescript
const fetcher = new MarketDataFetcher({
  apiBaseUrl: '/api/proxy/trading'
})

// Get real-time data
const data = await fetcher.getRealTimeData('USDCOP')

// Get candlesticks
const candles = await fetcher.getCandlestickData('USDCOP', '5m', undefined, undefined, 100)

// Get stats
const stats = await fetcher.getSymbolStats('USDCOP')

// Check market status
const isOpen = await fetcher.isMarketOpen()

// Subscribe to polling
const unsubscribe = fetcher.subscribeToPolling((data) => {
  console.log(data)
}, 2000)
```

### DataTransformer
```typescript
// Format price
DataTransformer.formatPrice(4123.45, 2) // "$4,123.45"

// Format volume
DataTransformer.formatVolume(1500000) // "1.50M"

// Format timestamp
DataTransformer.formatTimestamp(Date.now()) // "17 dic 2025, 10:30"

// Format percentage
DataTransformer.formatPercent(12.5, 2) // "12.50%"

// Convert timeframe
DataTransformer.timeframeToMinutes('5m') // 5
DataTransformer.minutesToTimeframe(60) // "1h"

// Normalize symbol
DataTransformer.normalizeSymbol('usdcop') // "USDCOP"
```

### StatisticsCalculator
```typescript
// Price change
StatisticsCalculator.calculatePriceChange(4100, 4050)
// => { change: 50, changePercent: 1.23, isPositive: true }

// Moving averages
StatisticsCalculator.calculateSMA([100, 102, 104, 106], 4) // 103
StatisticsCalculator.calculateEMA([100, 102, 104, 106], 4) // ~104.8

// Technical indicators
StatisticsCalculator.calculateRSI(prices, 14) // 0-100
StatisticsCalculator.calculateBollingerBands(prices, 20, 2)
// => { upper: 105.5, middle: 102.0, lower: 98.5 }

// Volatility
StatisticsCalculator.calculateVolatility(candles) // 0.0123

// Statistics from candles
StatisticsCalculator.calculatePriceStats(candles)
StatisticsCalculator.calculateVolumeStats(candles)
StatisticsCalculator.calculate24hStats(candles)

// VWAP
StatisticsCalculator.calculateVWAP(candles) // 4123.45
```

---

## Common Patterns

### Pattern 1: Real-Time Price Hook
```typescript
import { MarketDataFetcher } from '@/lib/services/market-data'
import { useState, useEffect, useMemo } from 'react'

export function useRealTimePrice(symbol: string) {
  const [price, setPrice] = useState(0)

  const fetcher = useMemo(
    () => new MarketDataFetcher({ apiBaseUrl: '/api/proxy/trading' }),
    []
  )

  useEffect(() => {
    return fetcher.subscribeToPolling((data) => {
      if (data.symbol === symbol) {
        setPrice(data.price)
      }
    }, 2000)
  }, [symbol, fetcher])

  return price
}
```

### Pattern 2: Formatted Price Display
```typescript
import { DataTransformer } from '@/lib/services/market-data'

export function PriceDisplay({ price }: { price: number }) {
  return <div>{DataTransformer.formatPrice(price, 2)}</div>
}
```

### Pattern 3: Statistics Dashboard
```typescript
import { MarketDataFetcher, StatisticsCalculator } from '@/lib/services/market-data'

export async function getMarketStats(symbol: string) {
  const fetcher = new MarketDataFetcher({ apiBaseUrl: '/api/proxy/trading' })
  const candles = await fetcher.getCandlestickData(symbol, '5m', undefined, undefined, 288)

  return {
    price: StatisticsCalculator.calculate24hStats(candles.data),
    stats: StatisticsCalculator.calculatePriceStats(candles.data),
    volume: StatisticsCalculator.calculateVolumeStats(candles.data),
  }
}
```

---

## Type Definitions

### MarketDataPoint
```typescript
interface MarketDataPoint {
  symbol: string
  price: number
  timestamp: number
  volume: number
  bid?: number
  ask?: number
  source?: string
}
```

### CandlestickData
```typescript
interface CandlestickData {
  time: number
  open: number
  high: number
  low: number
  close: number
  volume: number
}
```

### SymbolStats
```typescript
interface SymbolStats {
  symbol: string
  price: number
  open_24h: number
  high_24h: number
  low_24h: number
  volume_24h: number
  change_24h: number
  change_percent_24h: number
  spread: number
  timestamp: string
  source: string
}
```

---

## Configuration

### Environment Variables
```env
NEXT_PUBLIC_WS_URL=ws://localhost:8082
```

### API Base URL
```typescript
// Client-side
const fetcher = new MarketDataFetcher({
  apiBaseUrl: '/api/proxy/trading'
})

// Server-side
const fetcher = new MarketDataFetcher({
  apiBaseUrl: 'http://localhost:8000/api'
})
```

---

## Error Handling

### Graceful Fallbacks
```typescript
// WebSocket fails → Polling
// REST API fails → Historical data
// Stats endpoint fails → Calculate from candlesticks
// Market closed → Historical fallback

try {
  const data = await fetcher.getRealTimeData('USDCOP')
} catch (error) {
  // Error logged automatically
  // Fallback data returned when possible
}
```

### Custom Error Handling
```typescript
const fetcher = new MarketDataFetcher({ apiBaseUrl: '/api' })

try {
  const data = await fetcher.getRealTimeData('INVALID')
} catch (error) {
  if (error.message.includes('unavailable')) {
    // Handle data unavailable
  }
}
```

---

## Testing

### Mock Fetcher
```typescript
import { MarketDataFetcher } from '@/lib/services/market-data'

jest.mock('@/lib/services/market-data/MarketDataFetcher')

const mockFetcher = MarketDataFetcher as jest.Mocked<typeof MarketDataFetcher>
mockFetcher.prototype.getRealTimeData.mockResolvedValue([{
  symbol: 'USDCOP',
  price: 4123.45,
  timestamp: Date.now(),
  volume: 1000
}])
```

### Test Utilities
```typescript
describe('DataTransformer', () => {
  it('formats prices', () => {
    expect(DataTransformer.formatPrice(4123.45)).toContain('4,123')
  })
})

describe('StatisticsCalculator', () => {
  it('calculates SMA', () => {
    expect(StatisticsCalculator.calculateSMA([100, 102, 104], 3)).toBe(102)
  })
})
```

---

## Performance Tips

### 1. Reuse Instances
```typescript
// ✅ Good - reuse instance
const fetcher = useMemo(() => new MarketDataFetcher({ apiBaseUrl: '/api' }), [])

// ❌ Bad - creates new instance every render
const fetcher = new MarketDataFetcher({ apiBaseUrl: '/api' })
```

### 2. Import Only What You Need
```typescript
// ✅ Good - tree-shakeable
import { DataTransformer } from '@/lib/services/market-data'

// ❌ Less optimal - imports everything
import { MarketDataService } from '@/lib/services/market-data-service'
```

### 3. Use Memoization
```typescript
const stats = useMemo(
  () => StatisticsCalculator.calculatePriceStats(candles),
  [candles]
)
```

---

## Documentation Files

- **README.md** - Complete API documentation
- **MIGRATION_EXAMPLES.md** - Real-world migration examples
- **ARCHITECTURE.md** - Architecture deep dive
- **REFACTORING_SUMMARY.md** - Refactoring overview
- **QUICK_REFERENCE.md** - This file

---

## Support

For questions or issues:
1. Check [README.md](./README.md) for detailed documentation
2. Check [MIGRATION_EXAMPLES.md](./MIGRATION_EXAMPLES.md) for examples
3. Check [ARCHITECTURE.md](./ARCHITECTURE.md) for design details
4. Review test files in `__tests__/` for usage examples
