# Market Data Services - Modular Architecture

This directory contains the refactored Market Data services following the **Single Responsibility Principle (SRP)**.

## Module Structure

### 1. `types.ts`
**Responsibility**: Type definitions
- Shared TypeScript interfaces and types
- No logic, only type declarations

### 2. `WebSocketConnector.ts`
**Responsibility**: WebSocket lifecycle management
- Connect/disconnect WebSocket connections
- Handle reconnection logic
- Manage message subscriptions
- Handle WebSocket events (open, close, error, message)

**Usage Example**:
```typescript
import { WebSocketConnector } from '@/lib/services/market-data'

const connector = new WebSocketConnector({
  url: 'ws://localhost:8082',
  reconnectInterval: 5000,
  autoReconnect: true
})

// Connect to WebSocket
connector.connect('USDCOP')

// Subscribe to updates
connector.addSubscriber((data) => {
  console.log('Received:', data)
})

// Disconnect when done
connector.disconnect()
```

### 3. `MarketDataFetcher.ts`
**Responsibility**: REST API data retrieval
- Fetch real-time price data
- Fetch historical candlestick data
- Fetch symbol statistics
- Check API health
- Handle market open/close status
- Polling-based subscriptions

**Usage Example**:
```typescript
import { MarketDataFetcher } from '@/lib/services/market-data'

const fetcher = new MarketDataFetcher({
  apiBaseUrl: '/api/proxy/trading'
})

// Get real-time data
const data = await fetcher.getRealTimeData('USDCOP')

// Get candlestick data
const candles = await fetcher.getCandlestickData('USDCOP', '5m', undefined, undefined, 100)

// Get symbol stats
const stats = await fetcher.getSymbolStats('USDCOP')

// Check if market is open
const isOpen = await fetcher.isMarketOpen()
```

### 4. `DataTransformer.ts`
**Responsibility**: Data formatting and transformation
- Format prices with currency
- Format numbers and percentages
- Format volumes (K, M, B abbreviations)
- Format timestamps and dates
- Parse and normalize timeframes
- Utility transformations

**Usage Example**:
```typescript
import { DataTransformer } from '@/lib/services/market-data'

// Format price
const formatted = DataTransformer.formatPrice(4123.45, 2)
// => "$4,123.45"

// Format volume
const volume = DataTransformer.formatVolume(1500000)
// => "1.50M"

// Format timestamp
const date = DataTransformer.formatTimestamp(Date.now())
// => "17 dic 2025, 10:30:45"

// Convert timeframe
const minutes = DataTransformer.timeframeToMinutes('5m')
// => 5
```

### 5. `StatisticsCalculator.ts`
**Responsibility**: Market statistics calculations
- Calculate price changes and percentages
- Calculate moving averages (SMA, EMA)
- Calculate technical indicators (RSI, Bollinger Bands, ATR)
- Calculate volatility
- Calculate VWAP (Volume Weighted Average Price)
- Aggregate statistics

**Usage Example**:
```typescript
import { StatisticsCalculator } from '@/lib/services/market-data'

// Calculate price change
const change = StatisticsCalculator.calculatePriceChange(4100, 4050)
// => { change: 50, changePercent: 1.23, isPositive: true }

// Calculate SMA
const prices = [100, 102, 101, 103, 105]
const sma = StatisticsCalculator.calculateSMA(prices, 5)
// => 102.2

// Calculate RSI
const rsi = StatisticsCalculator.calculateRSI(prices, 14)

// Calculate Bollinger Bands
const bb = StatisticsCalculator.calculateBollingerBands(prices, 20, 2)
// => { upper: 105.5, middle: 102.0, lower: 98.5 }

// Calculate 24h stats
const stats = StatisticsCalculator.calculate24hStats(candles)
```

### 6. `index.ts`
**Responsibility**: Unified facade and exports
- Re-exports all modules and types
- Provides backwards-compatible `MarketDataService` class
- Maintains original API surface

## Migration Guide

### Option 1: Use the Backwards-Compatible Facade (Easiest)
No changes needed! The original `MarketDataService` class still works:

```typescript
import { MarketDataService } from '@/lib/services/market-data-service'

// All original methods still work
const data = await MarketDataService.getRealTimeData('USDCOP')
const formatted = MarketDataService.formatPrice(4123.45)
```

### Option 2: Use the New Modular Structure (Recommended)
For better tree-shaking and clearer separation of concerns:

```typescript
// Import only what you need
import {
  MarketDataFetcher,
  DataTransformer,
  StatisticsCalculator
} from '@/lib/services/market-data'

const fetcher = new MarketDataFetcher({ apiBaseUrl: '/api/proxy/trading' })
const data = await fetcher.getRealTimeData('USDCOP')
const formatted = DataTransformer.formatPrice(data[0].price)
```

## Benefits of the Refactoring

### Before (420+ lines, 6 mixed responsibilities)
- WebSocket management
- REST API calls
- Data formatting
- Statistics calculations
- Error handling
- All in one class

### After (Separated into 5 focused modules)
1. **Testability**: Each module can be tested independently
2. **Maintainability**: Changes to one responsibility don't affect others
3. **Reusability**: Modules can be used independently
4. **Tree-shaking**: Import only what you need
5. **Type Safety**: Better TypeScript support with focused interfaces
6. **Debugging**: Scoped loggers make debugging easier

## Key Improvements

1. **No Hardcoded Symbols**: All methods accept `symbol` parameter (defaults removed)
2. **Proper Logging**: Uses centralized logger with scoped contexts
3. **Better Error Handling**: Each module handles its own errors appropriately
4. **Type Safety**: Proper TypeScript interfaces for all parameters and returns
5. **Backwards Compatibility**: Original API maintained via facade pattern
6. **Configuration**: Services are configurable via constructor parameters

## Environment Variables

```env
NEXT_PUBLIC_WS_URL=ws://localhost:8082
```

## Architecture Principles Applied

- **Single Responsibility Principle**: Each class has one reason to change
- **Open/Closed Principle**: Open for extension, closed for modification
- **Dependency Inversion**: Depend on abstractions (interfaces), not concretions
- **Facade Pattern**: Simplified interface to complex subsystem
- **Strategy Pattern**: Different fetching strategies (WebSocket, REST, Polling)

## Testing

Each module can be tested independently:

```typescript
// Test DataTransformer
describe('DataTransformer', () => {
  it('should format prices correctly', () => {
    expect(DataTransformer.formatPrice(4123.45, 2)).toBe('$4,123.45')
  })
})

// Test StatisticsCalculator
describe('StatisticsCalculator', () => {
  it('should calculate SMA correctly', () => {
    const prices = [100, 102, 101, 103, 105]
    expect(StatisticsCalculator.calculateSMA(prices, 5)).toBe(102.2)
  })
})

// Test MarketDataFetcher with mocks
describe('MarketDataFetcher', () => {
  it('should fetch real-time data', async () => {
    const fetcher = new MarketDataFetcher({ apiBaseUrl: 'http://mock' })
    // Mock fetch...
  })
})
```

## File Structure

```
lib/services/market-data/
├── README.md                    # This file
├── types.ts                     # Type definitions
├── WebSocketConnector.ts        # WebSocket lifecycle (195 lines)
├── MarketDataFetcher.ts         # REST API fetching (295 lines)
├── DataTransformer.ts           # Data formatting (200 lines)
├── StatisticsCalculator.ts      # Statistics (290 lines)
└── index.ts                     # Facade & exports (180 lines)

lib/services/
└── market-data-service.ts       # Backwards compatibility layer (23 lines)
```

## Further Reading

- [SOLID Principles](https://en.wikipedia.org/wiki/SOLID)
- [Single Responsibility Principle](https://en.wikipedia.org/wiki/Single-responsibility_principle)
- [Facade Pattern](https://refactoring.guru/design-patterns/facade)
# Market Data Services - Architecture

## Overview

This document describes the architecture of the refactored Market Data services, following SOLID principles.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Client Applications                           │
│  (React Components, Hooks, API Routes, etc.)                    │
└────────────┬────────────────────────────────────┬───────────────┘
             │                                    │
             │ Option 1: Use Facade              │ Option 2: Direct
             │                                    │
             v                                    v
┌────────────────────────────┐     ┌────────────────────────────────┐
│   MarketDataService        │     │   Individual Modules           │
│   (Facade Pattern)         │     │                                │
│                            │     │  ┌──────────────────────────┐  │
│  - Backwards compatible    │     │  │ WebSocketConnector       │  │
│  - Delegates to modules    │     │  │ - connect()              │  │
│  - Simplified API          │     │  │ - disconnect()           │  │
└──────────┬─────────────────┘     │  │ - addSubscriber()        │  │
           │                       │  └──────────────────────────┘  │
           │ Delegates to          │                                │
           │                       │  ┌──────────────────────────┐  │
           v                       │  │ MarketDataFetcher        │  │
┌──────────────────────────────┐  │  │ - getRealTimeData()      │  │
│   Specialized Modules        │  │  │ - getCandlestickData()   │  │
│                              │  │  │ - getSymbolStats()       │  │
│  ┌────────────────────────┐ │  │  └──────────────────────────┘  │
│  │ WebSocketConnector     │ │  │                                │
│  │ (WebSocket Lifecycle)  │ │  │  ┌──────────────────────────┐  │
│  └────────────────────────┘ │  │  │ DataTransformer          │  │
│                              │  │  │ - formatPrice()          │  │
│  ┌────────────────────────┐ │  │  │ - formatVolume()         │  │
│  │ MarketDataFetcher      │ │  │  │ - formatTimestamp()      │  │
│  │ (REST API Calls)       │ │  │  └──────────────────────────┘  │
│  └────────────────────────┘ │  │                                │
│                              │  │  ┌──────────────────────────┐  │
│  ┌────────────────────────┐ │  │  │ StatisticsCalculator     │  │
│  │ DataTransformer        │ │  │  │ - calculatePriceChange() │  │
│  │ (Formatting)           │ │  │  │ - calculateSMA()         │  │
│  └────────────────────────┘ │  │  │ - calculateRSI()         │  │
│                              │  │  └──────────────────────────┘  │
│  ┌────────────────────────┐ │  │                                │
│  │ StatisticsCalculator   │ │  └────────────────────────────────┘
│  │ (Calculations)         │ │
│  └────────────────────────┘ │
└──────────────────────────────┘
           │
           v
┌──────────────────────────────┐
│   Shared Dependencies        │
│                              │
│  ┌────────────────────────┐ │
│  │ Logger (Scoped)        │ │
│  │ - debug(), info()      │ │
│  │ - warn(), error()      │ │
│  └────────────────────────┘ │
│                              │
│  ┌────────────────────────┐ │
│  │ Type Definitions       │ │
│  │ - MarketDataPoint      │ │
│  │ - CandlestickData      │ │
│  │ - SymbolStats          │ │
│  └────────────────────────┘ │
└──────────────────────────────┘
```

---

## Module Responsibilities

### 1. WebSocketConnector
**Single Responsibility**: WebSocket Lifecycle Management

```
Responsibilities:
├── Connect to WebSocket server
├── Handle connection events (open, close, error)
├── Implement reconnection logic
├── Manage message subscriptions
├── Parse incoming messages
└── Notify subscribers

Does NOT:
├── ❌ Fetch data from REST APIs
├── ❌ Format or transform data
├── ❌ Calculate statistics
└── ❌ Store or cache data
```

### 2. MarketDataFetcher
**Single Responsibility**: REST API Data Retrieval

```
Responsibilities:
├── Fetch real-time market data
├── Fetch historical candlestick data
├── Fetch symbol statistics
├── Check API health
├── Check market open/close status
└── Handle fallback data sources

Does NOT:
├── ❌ Manage WebSocket connections
├── ❌ Format prices for display
├── ❌ Calculate technical indicators
└── ❌ Transform data structure
```

### 3. DataTransformer
**Single Responsibility**: Data Formatting & Transformation

```
Responsibilities:
├── Format prices with currency
├── Format numbers and percentages
├── Format volumes (K, M, B)
├── Format timestamps and dates
├── Parse timeframes
├── Normalize symbols
└── Utility transformations

Does NOT:
├── ❌ Fetch data from APIs
├── ❌ Calculate statistics
├── ❌ Manage connections
└── ❌ Store data
```

### 4. StatisticsCalculator
**Single Responsibility**: Market Statistics & Calculations

```
Responsibilities:
├── Calculate price changes
├── Calculate moving averages (SMA, EMA)
├── Calculate technical indicators (RSI, BB, ATR)
├── Calculate volatility
├── Calculate VWAP
├── Aggregate statistics
└── Statistical functions

Does NOT:
├── ❌ Fetch data from APIs
├── ❌ Format data for display
├── ❌ Manage connections
└── ❌ Transform data structure
```

---

## Data Flow

### Real-Time Data Flow (WebSocket)

```
1. Client subscribes to updates
   └─> WebSocketConnector.connect(symbol)
       └─> Opens WebSocket connection
           └─> Sends subscribe message
               └─> Receives market data messages
                   └─> Parses JSON
                       └─> Transforms to MarketDataPoint
                           └─> Notifies all subscribers
                               └─> Client receives update
```

### REST API Data Flow

```
1. Client requests data
   └─> MarketDataFetcher.getRealTimeData(symbol)
       └─> Fetch from REST endpoint
           └─> Check response status
               ├─> Success: Transform response
               │   └─> Return MarketDataPoint[]
               └─> Error/Market Closed: Fallback
                   └─> getHistoricalFallback(symbol)
                       └─> Fetch candlestick data
                           └─> Return latest as MarketDataPoint[]
```

### Statistics Calculation Flow

```
1. Client needs statistics
   └─> MarketDataFetcher.getSymbolStats(symbol)
       └─> Try stats endpoint
           ├─> Success: Return stats
           └─> Failure: Fallback to calculation
               └─> getCandlestickData(symbol, '5m', ..., 288)
                   └─> StatisticsCalculator.calculate24hStats(candles)
                       └─> Return calculated SymbolStats
```

---

## Design Patterns Applied

### 1. Facade Pattern
```typescript
// MarketDataService acts as a facade
// Simplifies complex subsystem for backwards compatibility

class MarketDataService {
  static formatPrice() {
    return DataTransformer.formatPrice()  // Delegates
  }

  static calculatePriceChange() {
    return StatisticsCalculator.calculatePriceChange()  // Delegates
  }
}
```

### 2. Strategy Pattern
```typescript
// Different data fetching strategies
- WebSocket (real-time streaming)
- REST API (on-demand fetch)
- Polling (fallback for WebSocket)
- Historical fallback (when market closed)
```

### 3. Observer Pattern
```typescript
// WebSocketConnector notifies subscribers
class WebSocketConnector {
  private subscribers: Array<(data) => void> = []

  addSubscriber(callback) {
    this.subscribers.push(callback)
  }

  private notifySubscribers(data) {
    this.subscribers.forEach(callback => callback(data))
  }
}
```

### 4. Singleton Pattern (Logger)
```typescript
// Centralized logger instance
export const logger = new Logger()

// Scoped loggers for each module
const wsLogger = createLogger('WebSocketConnector')
const fetcherLogger = createLogger('MarketDataFetcher')
```

---

## SOLID Principles

### Single Responsibility Principle (SRP) ✅
- Each class has ONE reason to change
- WebSocketConnector: Changes when WebSocket protocol changes
- MarketDataFetcher: Changes when API endpoints change
- DataTransformer: Changes when formatting requirements change
- StatisticsCalculator: Changes when calculation methods change

### Open/Closed Principle (OCP) ✅
- Modules are open for extension (inheritance)
- Closed for modification (existing code stable)
```typescript
// Example: Extend fetcher with caching
class CachedMarketDataFetcher extends MarketDataFetcher {
  // Add caching layer without modifying original
}
```

### Liskov Substitution Principle (LSP) ✅
- Subclasses can replace base classes without breaking
- All modules have clear contracts (interfaces)

### Interface Segregation Principle (ISP) ✅
- No client forced to depend on unused methods
- Clients import only what they need
```typescript
// Import only what you need
import { DataTransformer } from '@/lib/services/market-data'
// Don't need fetcher, stats, or websocket
```

### Dependency Inversion Principle (DIP) ✅
- Depend on abstractions (types/interfaces)
- Not concrete implementations
```typescript
// Depend on MarketDataPoint interface, not concrete class
function processData(data: MarketDataPoint) {
  // Works with any implementation
}
```

---

## Error Handling Strategy

### 1. Module-Level Error Handling
Each module handles its own errors:

```typescript
// WebSocketConnector
try {
  this.websocket = new WebSocket(url)
} catch (error) {
  logger.error('Failed to connect:', error)
  return null
}

// MarketDataFetcher
try {
  const response = await fetch(url)
} catch (error) {
  logger.error('Fetch failed:', error)
  return this.getHistoricalFallback(symbol)
}
```

### 2. Graceful Degradation
```
WebSocket fails → Fall back to Polling
REST API fails → Fall back to Historical data
Stats endpoint fails → Calculate from candlesticks
No data available → Return default values
```

### 3. Error Propagation
```typescript
// Let critical errors bubble up
throw new Error('Real data unavailable')

// Handle recoverable errors
logger.warn('Using fallback data')
return fallbackData
```

---

## Testing Strategy

### Unit Tests (Isolated)
```typescript
// Test each module independently
describe('DataTransformer', () => {
  it('should format prices', () => {
    expect(DataTransformer.formatPrice(4123.45)).toBe('$4,123.45')
  })
})

describe('StatisticsCalculator', () => {
  it('should calculate SMA', () => {
    expect(StatisticsCalculator.calculateSMA([100, 102, 104], 3)).toBe(102)
  })
})
```

### Integration Tests (Combined)
```typescript
// Test modules working together
describe('MarketDataService', () => {
  it('should fetch and format data', async () => {
    const data = await MarketDataService.getRealTimeData('USDCOP')
    const formatted = MarketDataService.formatPrice(data[0].price)
    expect(formatted).toContain('$')
  })
})
```

### E2E Tests (Full Flow)
```typescript
// Test complete user scenarios
it('should display real-time prices', async () => {
  render(<PriceDisplay symbol="USDCOP" />)
  await waitFor(() => {
    expect(screen.getByText(/\$/)).toBeInTheDocument()
  })
})
```

---

## Performance Considerations

### 1. Lazy Initialization
```typescript
// Don't create instances until needed
private static fetcher: MarketDataFetcher | null = null

private static getFetcher() {
  if (!this.fetcher) {
    this.fetcher = new MarketDataFetcher(config)
  }
  return this.fetcher
}
```

### 2. Tree Shaking
```typescript
// Import only what you need
import { DataTransformer } from '@/lib/services/market-data'
// WebSocket, Fetcher, Stats won't be in bundle
```

### 3. Memoization
```typescript
// Cache expensive calculations
const cachedSMA = useMemo(
  () => StatisticsCalculator.calculateSMA(prices, 20),
  [prices]
)
```

---

## Future Enhancements

### Potential Improvements
1. **Caching Layer**: Add LRU cache for API responses
2. **Request Batching**: Batch multiple symbol requests
3. **Compression**: WebSocket message compression
4. **State Management**: Integration with Redux/Zustand
5. **Persistence**: Local storage for offline support
6. **Rate Limiting**: Client-side rate limit handling
7. **Retry Logic**: Exponential backoff for failed requests
8. **Metrics**: Performance monitoring and analytics

### Extension Points
```typescript
// Easy to extend without modifying core
class EnhancedMarketDataFetcher extends MarketDataFetcher {
  private cache = new LRUCache()

  async getCandlestickData(...args) {
    const cached = this.cache.get(key)
    if (cached) return cached

    const data = await super.getCandlestickData(...args)
    this.cache.set(key, data)
    return data
  }
}
```

---

## Comparison: Before vs After

### Before (Monolithic - 420 lines)
```
✗ 6 responsibilities mixed together
✗ Hard to test individual functions
✗ Hard to mock for testing
✗ Tight coupling
✗ Large bundle size (import all or nothing)
✗ Hardcoded 'USDCOP' symbol
✗ Console.log everywhere
```

### After (Modular - 1,399 lines across 6 files)
```
✓ Single responsibility per module
✓ Easy to test each module
✓ Easy to mock dependencies
✓ Loose coupling
✓ Tree-shakeable (import only what you need)
✓ Configurable symbols
✓ Centralized, scoped logging
✓ Better TypeScript support
✓ Extensible via inheritance
✓ Backwards compatible
```

---

## Conclusion

The refactored architecture provides:
- **Maintainability**: Easy to understand and modify
- **Testability**: Each module can be tested independently
- **Reusability**: Modules can be used in different contexts
- **Scalability**: Easy to add new features
- **Performance**: Tree-shaking reduces bundle size
- **Type Safety**: Better TypeScript intellisense
- **Backwards Compatibility**: No breaking changes

For detailed usage examples, see [MIGRATION_EXAMPLES.md](./MIGRATION_EXAMPLES.md)
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
