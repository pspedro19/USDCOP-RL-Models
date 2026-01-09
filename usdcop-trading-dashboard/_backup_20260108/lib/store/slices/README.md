# Trading Store Slices

This directory contains the refactored Trading Store following the **Interface Segregation Principle** (ISP).

## Architecture Overview

The monolithic `TradingStore` has been split into 5 focused slices, each with a single responsibility:

```
lib/store/slices/
├── marketDataSlice.ts    # Real-time market data & connection
├── signalsSlice.ts       # Trading signals from PPO-LSTM
├── positionsSlice.ts     # Current positions & PnL
├── tradesSlice.ts        # Trade history & pending orders
├── uiSlice.ts            # User preferences & display settings
└── index.ts              # Central exports
```

## Benefits of This Architecture

### 1. Interface Segregation Principle (ISP)
- Components only import what they need
- No forced dependencies on unused state/methods
- Clear separation of concerns

### 2. Performance Optimization
- Reduced re-renders (components only subscribe to relevant slices)
- Smaller bundle sizes (tree-shaking friendly)
- Better code splitting opportunities

### 3. Maintainability
- Easier to test individual slices
- Simpler to understand and modify
- Less coupling between features

### 4. Type Safety
- Strongly typed slice interfaces
- Better IDE autocomplete
- Compile-time error detection

## Usage Guide

### Recommended: Use Slice Hooks

Import only the slice you need for maximum performance:

```typescript
// ✅ GOOD: Only subscribes to market data changes
import { useMarketDataSlice } from '@/lib/store/trading-store'

function PriceDisplay() {
  const { marketData, isConnected } = useMarketDataSlice()

  return <div>{marketData.price}</div>
}
```

```typescript
// ✅ GOOD: Only subscribes to signal changes
import { useSignalsSlice } from '@/lib/store/trading-store'

function SignalIndicator() {
  const { latestSignal, isLoading } = useSignalsSlice()

  return <div>{latestSignal?.action}</div>
}
```

### Available Slice Hooks

```typescript
// Market Data
const marketDataSlice = useMarketDataSlice()
// Returns: { marketData, isConnected, lastUpdate, updateMarketData, setConnectionStatus, getLatestPrice }

// Signals
const signalsSlice = useSignalsSlice()
// Returns: { signals, latestSignal, isLoading, error, addSignal, setSignals, clearSignals, ... }

// Positions
const positionsSlice = usePositionsSlice()
// Returns: { positions, activePosition, totalPnl, totalUnrealizedPnl, setPositions, ... }

// Trades
const tradesSlice = useTradesSlice()
// Returns: { trades, todayTrades, pendingOrders, addTrade, setTrades }

// UI
const uiSlice = useUISlice()
// Returns: { selectedTimeframe, showVolume, theme, chartHeight, setTimeframe, ... }
```

### When to Use Multiple Slices

If a component needs data from multiple slices, you can use multiple hooks:

```typescript
function TradingDashboard() {
  const { marketData, isConnected } = useMarketDataSlice()
  const { latestSignal } = useSignalsSlice()
  const { activePosition } = usePositionsSlice()

  // Component re-renders only when any of these slices change
  return (...)
}
```

### Backwards Compatibility

Old hooks are still available but deprecated:

```typescript
// ⚠️ DEPRECATED: Use useMarketDataSlice() instead
const marketData = useMarketData()
const isConnected = useConnectionStatus()
const latestSignal = useLatestSignal()
const activePosition = useActivePosition()
const todayTrades = useTodayTrades()
const ui = useUIState()
```

### Using Granular Selectors

For even better performance, use specific selectors:

```typescript
import { useTradingStore, selectLatestPrice } from '@/lib/store/trading-store'

function PriceDisplay() {
  // Only re-renders when price changes (not on any market data change)
  const price = useTradingStore(selectLatestPrice)

  return <div>{price}</div>
}
```

Available selectors:
- `selectMarketData`, `selectIsConnected`, `selectLatestPrice`
- `selectLatestSignal`, `selectSignals`, `selectSignalLoading`
- `selectActivePosition`, `selectPositions`, `selectTotalPnl`
- `selectTodayTrades`, `selectAllTrades`
- `selectTimeframe`, `selectTheme`, `selectIndicators`

### Full Store Access (Not Recommended)

If you absolutely need the full store:

```typescript
// ❌ AVOID: Subscribes to ALL state changes
import { useTradingStore } from '@/lib/store/trading-store'

function MyComponent() {
  const store = useTradingStore() // Re-renders on ANY state change!
  // ...
}
```

## Slice Details

### 1. Market Data Slice (`marketDataSlice.ts`)

**Responsibility:** Real-time price data and WebSocket connection status

**State:**
- `marketData`: OHLCV data, spread, volume, 24h changes
- `isConnected`: WebSocket connection status
- `lastUpdate`: Timestamp of last update

**Actions:**
- `updateMarketData(data)`: Update market data
- `setConnectionStatus(connected)`: Set connection status
- `getLatestPrice()`: Get current price

**Use Cases:**
- Price tickers
- Connection indicators
- Market stats widgets

---

### 2. Signals Slice (`signalsSlice.ts`)

**Responsibility:** Trading signals from the PPO-LSTM model

**State:**
- `signals`: Array of historical signals (max 100)
- `latestSignal`: Most recent signal
- `isLoading`: Signal loading state
- `error`: Error messages

**Actions:**
- `addSignal(signal)`: Add new signal
- `setSignals(signals)`: Replace all signals
- `clearSignals()`: Clear all signals
- `setSignalLoading(loading)`: Set loading state
- `setSignalError(error)`: Set error state

**Use Cases:**
- Signal displays
- Signal history charts
- Model output visualization

---

### 3. Positions Slice (`positionsSlice.ts`)

**Responsibility:** Current trading positions and PnL tracking

**State:**
- `positions`: Array of all positions
- `activePosition`: Currently active position
- `totalPnl`: Total realized PnL
- `totalUnrealizedPnl`: Total unrealized PnL

**Actions:**
- `setPositions(positions)`: Set all positions
- `updatePosition(id, updates)`: Update specific position
- `setActivePosition(position)`: Set active position
- `closePosition(id)`: Close a position
- `getPositionSide()`: Get current position side
- `getTotalPnl()`: Get total PnL

**Use Cases:**
- Position tables
- PnL displays
- Portfolio summary

---

### 4. Trades Slice (`tradesSlice.ts`)

**Responsibility:** Trade history and order tracking

**State:**
- `trades`: All historical trades (max 500)
- `todayTrades`: Today's trades only
- `pendingOrders`: Count of pending orders

**Actions:**
- `addTrade(trade)`: Add new trade
- `setTrades(trades)`: Replace all trades

**Use Cases:**
- Trade history tables
- Daily performance tracking
- Order book displays

---

### 5. UI Slice (`uiSlice.ts`)

**Responsibility:** User preferences and display settings

**State:**
- `selectedTimeframe`: Chart timeframe ('5m', '15m', etc.)
- `showVolume`: Volume chart visibility
- `showIndicators`: Technical indicators visibility
- `indicatorsEnabled`: Individual indicator toggles (EMA, BB, RSI, MACD)
- `theme`: Dark/light theme
- `chartHeight`: Chart height in pixels

**Actions:**
- `setTimeframe(timeframe)`: Change chart timeframe
- `toggleIndicator(indicator)`: Toggle specific indicator
- `setShowVolume(show)`: Toggle volume display
- `setShowIndicators(show)`: Toggle all indicators
- `setTheme(theme)`: Set theme
- `setChartHeight(height)`: Set chart height

**Use Cases:**
- Chart controls
- Settings panels
- Theme switchers

---

## Migration Guide

### Before (Monolithic Store)

```typescript
import { useTradingStore } from '@/lib/store/trading-store'

function MyComponent() {
  const store = useTradingStore()
  // Re-renders on ANY state change!

  return <div>{store.marketData.price}</div>
}
```

### After (Slice-Based)

```typescript
import { useMarketDataSlice } from '@/lib/store/trading-store'

function MyComponent() {
  const { marketData } = useMarketDataSlice()
  // Only re-renders on market data changes!

  return <div>{marketData.price}</div>
}
```

### Performance Impact

| Approach | Re-renders | Bundle Size | Type Safety |
|----------|-----------|-------------|-------------|
| Full Store | High (any change) | Large | Good |
| Slice Hook | Medium (slice changes) | Medium | Excellent |
| Granular Selector | Low (specific field) | Small | Excellent |

## Testing

Each slice can be tested independently:

```typescript
import { createMarketDataSlice } from '@/lib/store/slices/marketDataSlice'

describe('MarketDataSlice', () => {
  it('should update market data', () => {
    const store = create(createMarketDataSlice)

    store.getState().updateMarketData({ price: 4200 })

    expect(store.getState().marketData.price).toBe(4200)
  })
})
```

## Best Practices

1. **Use Slice Hooks First**: Start with `useMarketDataSlice()` etc.
2. **Granular Selectors for Primitives**: Use `selectLatestPrice` for single values
3. **Avoid Full Store**: Only use `useTradingStore()` when absolutely necessary
4. **Test Slices Independently**: Each slice is a separate unit
5. **Keep Slices Focused**: Don't add cross-cutting concerns

## Future Enhancements

Potential additions to consider:

- `alertsSlice.ts` - Price alerts and notifications
- `ordersSlice.ts` - Pending orders management
- `strategySlice.ts` - Strategy configuration
- `performanceSlice.ts` - Performance metrics and analytics
- `backtestSlice.ts` - Backtesting results

Each new feature should get its own slice following the same pattern.
