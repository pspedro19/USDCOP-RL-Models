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
