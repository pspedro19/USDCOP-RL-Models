# Trading Store - Quick Reference

## Import Cheat Sheet

```typescript
// Slice Hooks (Use these!)
import {
  useMarketDataSlice,   // Market data + connection
  useSignalsSlice,      // Trading signals
  usePositionsSlice,    // Positions + PnL
  useTradesSlice,       // Trade history
  useUISlice,           // UI preferences
} from '@/lib/store/trading-store'

// Granular Selectors (For single values)
import {
  useTradingStore,
  selectLatestPrice,    // Just the price
  selectIsConnected,    // Just connection status
  selectLatestSignal,   // Just latest signal
  selectActivePosition, // Just active position
  selectTotalPnl,       // Just total PnL
} from '@/lib/store/trading-store'

// Types
import type {
  MarketData,
  Timeframe,
  Theme,
} from '@/lib/store/trading-store'
```

## Common Patterns

### Pattern 1: Display Price
```typescript
const { marketData } = useMarketDataSlice()
<div>${marketData.price.toFixed(2)}</div>
```

### Pattern 2: Show Connection Status
```typescript
const { isConnected } = useMarketDataSlice()
<span>{isConnected ? '🟢' : '🔴'}</span>
```

### Pattern 3: Update Price (WebSocket)
```typescript
const { updateMarketData } = useMarketDataSlice()

websocket.onmessage = (e) => {
  const data = JSON.parse(e.data)
  updateMarketData({ price: data.price })
}
```

### Pattern 4: Display Latest Signal
```typescript
const { latestSignal } = useSignalsSlice()
<div>{latestSignal?.action}</div>
```

### Pattern 5: Add New Signal
```typescript
const { addSignal } = useSignalsSlice()
addSignal({ id: '1', action: 'BUY', confidence: 0.95, ... })
```

### Pattern 6: Show Active Position
```typescript
const { activePosition } = usePositionsSlice()
<div>{activePosition ? activePosition.side : 'FLAT'}</div>
```

### Pattern 7: Update Positions
```typescript
const { setPositions } = usePositionsSlice()
setPositions(newPositions) // Auto-calculates PnL
```

### Pattern 8: Show Total PnL
```typescript
const { getTotalPnl } = usePositionsSlice()
<div>${getTotalPnl().toFixed(2)}</div>
```

### Pattern 9: Display Today's Trades
```typescript
const { todayTrades } = useTradesSlice()
{todayTrades.map(trade => <TradeRow key={trade.id} trade={trade} />)}
```

### Pattern 10: Add New Trade
```typescript
const { addTrade } = useTradesSlice()
addTrade({ id: '1', side: 'BUY', price: 4200, ... })
```

### Pattern 11: Change Timeframe
```typescript
const { selectedTimeframe, setTimeframe } = useUISlice()
<select value={selectedTimeframe} onChange={(e) => setTimeframe(e.target.value)}>
  <option value="5m">5m</option>
  <option value="1h">1h</option>
</select>
```

### Pattern 12: Toggle Indicator
```typescript
const { indicatorsEnabled, toggleIndicator } = useUISlice()
<button onClick={() => toggleIndicator('ema')}>
  EMA {indicatorsEnabled.ema ? 'ON' : 'OFF'}
</button>
```

### Pattern 13: Multiple Slices
```typescript
const { marketData } = useMarketDataSlice()
const { latestSignal } = useSignalsSlice()
const { activePosition } = usePositionsSlice()

<Dashboard price={marketData.price} signal={latestSignal} position={activePosition} />
```

### Pattern 14: Granular Selector (Best Performance)
```typescript
const price = useTradingStore(selectLatestPrice)
<PriceTag>${price}</PriceTag> // Only re-renders when price changes
```

## State Structure

```typescript
// Market Data Slice
{
  marketData: {
    symbol: 'USDCOP',
    price: 4200.50,
    bid: 4200.00,
    ask: 4201.00,
    spread: 1.00,
    volume24h: 1000000,
    change24h: 50.25,
    changePercent24h: 1.21,
    high24h: 4250.00,
    low24h: 4150.00,
    timestamp: 1234567890
  },
  isConnected: true,
  lastUpdate: 1234567890
}

// Signals Slice
{
  signals: [{ id: '1', action: 'BUY', ... }],
  latestSignal: { id: '1', action: 'BUY', ... },
  isLoading: false,
  error: null
}

// Positions Slice
{
  positions: [{ id: '1', side: 'LONG', ... }],
  activePosition: { id: '1', side: 'LONG', ... },
  totalPnl: 1234.56,
  totalUnrealizedPnl: 123.45
}

// Trades Slice
{
  trades: [{ id: '1', side: 'BUY', ... }],
  todayTrades: [{ id: '1', side: 'BUY', ... }],
  pendingOrders: 0
}

// UI Slice
{
  selectedTimeframe: '5m',
  showVolume: true,
  showIndicators: true,
  indicatorsEnabled: { ema: true, bb: true, rsi: false, macd: false },
  theme: 'dark',
  chartHeight: 500
}
```

## Action Reference

### Market Data Actions
- `updateMarketData(data)` - Update market data
- `setConnectionStatus(connected)` - Set WebSocket connection status
- `getLatestPrice()` - Get current price

### Signals Actions
- `addSignal(signal)` - Add new signal (max 100 kept)
- `setSignals(signals)` - Replace all signals
- `clearSignals()` - Clear all signals
- `setSignalLoading(loading)` - Set loading state
- `setSignalError(error)` - Set error state

### Positions Actions
- `setPositions(positions)` - Set all positions (auto-calculates PnL)
- `updatePosition(id, updates)` - Update specific position
- `setActivePosition(position)` - Set active position
- `closePosition(id)` - Close position by ID
- `getPositionSide()` - Get current position side
- `getTotalPnl()` - Get total PnL (realized + unrealized)

### Trades Actions
- `addTrade(trade)` - Add new trade (max 500 kept)
- `setTrades(trades)` - Replace all trades

### UI Actions
- `setTimeframe(timeframe)` - Set chart timeframe
- `toggleIndicator(indicator)` - Toggle indicator on/off
- `setShowVolume(show)` - Show/hide volume chart
- `setShowIndicators(show)` - Show/hide all indicators
- `setTheme(theme)` - Set theme (dark/light)
- `setChartHeight(height)` - Set chart height in pixels

## Selector Reference

### Pre-built Selectors
```typescript
// Market Data
selectMarketData(state)       // Full market data object
selectIsConnected(state)      // Connection status
selectLatestPrice(state)      // Just the price

// Signals
selectLatestSignal(state)     // Latest signal
selectSignals(state)          // All signals array
selectSignalLoading(state)    // Loading state

// Positions
selectActivePosition(state)   // Active position
selectPositions(state)        // All positions array
selectTotalPnl(state)         // Total PnL

// Trades
selectTodayTrades(state)      // Today's trades
selectAllTrades(state)        // All trades array

// UI
selectTimeframe(state)        // Selected timeframe
selectTheme(state)            // Current theme
selectIndicators(state)       // Indicators enabled object
```

### Custom Selector
```typescript
// Create your own selector
const selectSpread = (state) => state.marketData.spread
const spread = useTradingStore(selectSpread)
```

## Performance Tips

### ✅ Good Performance
```typescript
// Use slice hook
const { marketData } = useMarketDataSlice()

// Use granular selector for single value
const price = useTradingStore(selectLatestPrice)

// Use multiple slice hooks
const { marketData } = useMarketDataSlice()
const { latestSignal } = useSignalsSlice()
```

### ❌ Poor Performance
```typescript
// Don't use full store
const store = useTradingStore() // Re-renders on ANY change!

// Don't destructure everything
const { marketData, signals, positions, ... } = useTradingStore()
```

## Testing Snippets

### Test Slice
```typescript
import { create } from 'zustand'
import { createMarketDataSlice } from '@/lib/store/slices/marketDataSlice'

const store = create(createMarketDataSlice)
store.getState().updateMarketData({ price: 4200 })
expect(store.getState().marketData.price).toBe(4200)
```

### Test Component
```typescript
import { renderHook } from '@testing-library/react'
import { useMarketDataSlice } from '@/lib/store/trading-store'

const { result } = renderHook(() => useMarketDataSlice())
act(() => result.current.updateMarketData({ price: 4200 }))
expect(result.current.marketData.price).toBe(4200)
```

## TypeScript Types

```typescript
// Market Data
type MarketData = {
  symbol: string
  price: number
  bid: number
  ask: number
  spread: number
  volume24h: number
  change24h: number
  changePercent24h: number
  high24h: number
  low24h: number
  timestamp: number
}

// UI
type Timeframe = '5m' | '15m' | '30m' | '1h' | '4h' | '1d'
type Theme = 'dark' | 'light'
type IndicatorsEnabled = {
  ema: boolean
  bb: boolean
  rsi: boolean
  macd: boolean
}
```

## DevTools

1. Install Redux DevTools extension
2. Open browser DevTools
3. Look for "TradingStore" tab
4. Inspect state changes in real-time
5. Time-travel debug

## Common Mistakes

### ❌ Mistake 1: Full store subscription
```typescript
const store = useTradingStore()
```
**Fix:** Use slice hook instead
```typescript
const { marketData } = useMarketDataSlice()
```

### ❌ Mistake 2: Nested object mutation
```typescript
marketData.price = 4200 // Don't mutate!
```
**Fix:** Use update action
```typescript
updateMarketData({ price: 4200 })
```

### ❌ Mistake 3: Forgetting to call action
```typescript
const { addSignal } = useSignalsSlice()
// Later...
addSignal // Wrong! This is just the function reference
```
**Fix:** Call the action
```typescript
addSignal({ id: '1', action: 'BUY', ... })
```

### ❌ Mistake 4: Using wrong slice
```typescript
const { marketData } = useSignalsSlice() // Wrong slice!
```
**Fix:** Use correct slice
```typescript
const { marketData } = useMarketDataSlice()
```

## Need Help?

- [Full README](./README.md)
- [Slice Details](./slices/README.md)
- [Architecture](./ARCHITECTURE.md)
- [Migration Guide](./MIGRATION_GUIDE.md)
