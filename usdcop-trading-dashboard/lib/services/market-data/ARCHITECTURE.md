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
