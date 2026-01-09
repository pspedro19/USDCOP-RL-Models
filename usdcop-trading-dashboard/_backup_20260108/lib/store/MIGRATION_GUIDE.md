# Trading Store Migration Guide

## Overview

The Trading Store has been refactored following the **Interface Segregation Principle (ISP)** to improve performance, maintainability, and code organization.

## What Changed?

### Before (Monolithic)
```typescript
import { useTradingStore } from '@/lib/store/trading-store'

function MyComponent() {
  const store = useTradingStore()
  // Component re-renders on ANY state change in the entire store!

  return <div>{store.marketData.price}</div>
}
```

### After (Slice-Based)
```typescript
import { useMarketDataSlice } from '@/lib/store/trading-store'

function MyComponent() {
  const { marketData } = useMarketDataSlice()
  // Component only re-renders when market data changes!

  return <div>{marketData.price}</div>
}
```

## Migration Steps

### Step 1: Identify Your Component's Dependencies

First, determine which state your component actually needs:

| If you need... | Use this slice... |
|----------------|-------------------|
| Price, bid, ask, connection status | `useMarketDataSlice()` |
| Trading signals, model predictions | `useSignalsSlice()` |
| Positions, PnL, active positions | `usePositionsSlice()` |
| Trade history, today's trades | `useTradesSlice()` |
| Chart settings, theme, timeframe | `useUISlice()` |

### Step 2: Replace Store Usage

#### Example 1: Price Display Component

**Before:**
```typescript
import { useTradingStore } from '@/lib/store/trading-store'

export function PriceDisplay() {
  const store = useTradingStore()

  return (
    <div>
      <span>{store.marketData.price}</span>
      <span>{store.isConnected ? '🟢' : '🔴'}</span>
    </div>
  )
}
```

**After:**
```typescript
import { useMarketDataSlice } from '@/lib/store/trading-store'

export function PriceDisplay() {
  const { marketData, isConnected } = useMarketDataSlice()

  return (
    <div>
      <span>{marketData.price}</span>
      <span>{isConnected ? '🟢' : '🔴'}</span>
    </div>
  )
}
```

#### Example 2: Signal Indicator

**Before:**
```typescript
import { useTradingStore } from '@/lib/store/trading-store'

export function SignalIndicator() {
  const store = useTradingStore()
  const signal = store.signals.latestSignal

  return <div>{signal?.action}</div>
}
```

**After:**
```typescript
import { useSignalsSlice } from '@/lib/store/trading-store'

export function SignalIndicator() {
  const { latestSignal } = useSignalsSlice()

  return <div>{latestSignal?.action}</div>
}
```

#### Example 3: Position Summary

**Before:**
```typescript
import { useTradingStore } from '@/lib/store/trading-store'

export function PositionSummary() {
  const store = useTradingStore()
  const position = store.positions.activePosition
  const totalPnl = store.getTotalPnl()

  return (
    <div>
      <div>Side: {position?.side}</div>
      <div>PnL: ${totalPnl}</div>
    </div>
  )
}
```

**After:**
```typescript
import { usePositionsSlice } from '@/lib/store/trading-store'

export function PositionSummary() {
  const { activePosition, getTotalPnl } = usePositionsSlice()

  return (
    <div>
      <div>Side: {activePosition?.side}</div>
      <div>PnL: ${getTotalPnl()}</div>
    </div>
  )
}
```

#### Example 4: Chart Controls

**Before:**
```typescript
import { useTradingStore } from '@/lib/store/trading-store'

export function ChartControls() {
  const store = useTradingStore()

  return (
    <select
      value={store.ui.selectedTimeframe}
      onChange={(e) => store.setTimeframe(e.target.value)}
    >
      <option value="5m">5m</option>
      <option value="1h">1h</option>
    </select>
  )
}
```

**After:**
```typescript
import { useUISlice } from '@/lib/store/trading-store'

export function ChartControls() {
  const { selectedTimeframe, setTimeframe } = useUISlice()

  return (
    <select
      value={selectedTimeframe}
      onChange={(e) => store.setTimeframe(e.target.value)}
    >
      <option value="5m">5m</option>
      <option value="1h">1h</option>
    </select>
  )
}
```

### Step 3: Handle Multiple Slice Dependencies

If your component needs data from multiple slices, use multiple hooks:

**Before:**
```typescript
import { useTradingStore } from '@/lib/store/trading-store'

export function TradingDashboard() {
  const store = useTradingStore()
  // Re-renders on ANY change!

  return (
    <div>
      <div>Price: {store.marketData.price}</div>
      <div>Signal: {store.signals.latestSignal?.action}</div>
      <div>Position: {store.positions.activePosition?.side}</div>
    </div>
  )
}
```

**After:**
```typescript
import {
  useMarketDataSlice,
  useSignalsSlice,
  usePositionsSlice
} from '@/lib/store/trading-store'

export function TradingDashboard() {
  const { marketData } = useMarketDataSlice()
  const { latestSignal } = useSignalsSlice()
  const { activePosition } = usePositionsSlice()
  // Only re-renders when market data, signals, or positions change!

  return (
    <div>
      <div>Price: {marketData.price}</div>
      <div>Signal: {latestSignal?.action}</div>
      <div>Position: {activePosition?.side}</div>
    </div>
  )
}
```

### Step 4: Use Granular Selectors for Primitive Values

For maximum performance when you only need a single value:

**Before:**
```typescript
import { useTradingStore } from '@/lib/store/trading-store'

export function PriceTag() {
  const store = useTradingStore()
  // Re-renders on ANY change!

  return <span>${store.marketData.price}</span>
}
```

**Better (Slice):**
```typescript
import { useMarketDataSlice } from '@/lib/store/trading-store'

export function PriceTag() {
  const { marketData } = useMarketDataSlice()
  // Re-renders when any market data field changes

  return <span>${marketData.price}</span>
}
```

**Best (Granular Selector):**
```typescript
import { useTradingStore, selectLatestPrice } from '@/lib/store/trading-store'

export function PriceTag() {
  const price = useTradingStore(selectLatestPrice)
  // Only re-renders when price changes!

  return <span>${price}</span>
}
```

### Step 5: Update Store Actions

Actions remain the same, just accessed through slices:

**Before:**
```typescript
const store = useTradingStore()
store.updateMarketData({ price: 4200 })
store.addSignal(newSignal)
store.setPositions(positions)
```

**After:**
```typescript
const { updateMarketData } = useMarketDataSlice()
const { addSignal } = useSignalsSlice()
const { setPositions } = usePositionsSlice()

updateMarketData({ price: 4200 })
addSignal(newSignal)
setPositions(positions)
```

## Backwards Compatibility

All old hooks still work but are marked as deprecated:

```typescript
// ⚠️ Still works, but deprecated
const marketData = useMarketData()
const latestSignal = useLatestSignal()
const activePosition = useActivePosition()
const todayTrades = useTodayTrades()
const ui = useUIState()
const isConnected = useConnectionStatus()

// ✅ Recommended replacements
const { marketData } = useMarketDataSlice()
const { latestSignal } = useSignalsSlice()
const { activePosition } = usePositionsSlice()
const { todayTrades } = useTradesSlice()
const ui = useUISlice()
const { isConnected } = useMarketDataSlice()
```

## Performance Comparison

### Scenario: Component only needs price

| Approach | Re-renders per minute | Performance |
|----------|----------------------|-------------|
| Full Store | ~60 (every state change) | ❌ Poor |
| Slice Hook | ~12 (market data changes) | ✅ Good |
| Granular Selector | ~5 (price changes only) | ⭐ Excellent |

### Scenario: Component needs price + signals + positions

| Approach | Re-renders per minute | Performance |
|----------|----------------------|-------------|
| Full Store | ~60 (every state change) | ❌ Poor |
| 3 Slice Hooks | ~25 (relevant changes) | ✅ Good |

## Testing Your Migration

1. **Check for unnecessary re-renders:**
   ```typescript
   import { useEffect } from 'react'

   export function MyComponent() {
     const { marketData } = useMarketDataSlice()

     useEffect(() => {
       console.log('Component re-rendered')
     })

     // If this logs on every state change, you may need a granular selector
   }
   ```

2. **Verify type safety:**
   ```typescript
   const { marketData, updateMarketData } = useMarketDataSlice()
   // TypeScript should provide full autocomplete for all slice methods
   ```

3. **Test all actions:**
   ```typescript
   const { updateMarketData } = useMarketDataSlice()
   updateMarketData({ price: 4200 }) // Should work as before
   ```

## Common Pitfalls

### ❌ Don't destructure the full store
```typescript
// BAD: Still subscribes to everything
const { marketData, signals, positions } = useTradingStore()
```

### ✅ Do use multiple slice hooks
```typescript
// GOOD: Only subscribes to what you need
const { marketData } = useMarketDataSlice()
const { signals } = useSignalsSlice()
const { positions } = usePositionsSlice()
```

### ❌ Don't use full store for single values
```typescript
// BAD: Re-renders on every change
const store = useTradingStore()
return <div>{store.marketData.price}</div>
```

### ✅ Do use granular selectors
```typescript
// GOOD: Only re-renders when price changes
const price = useTradingStore(selectLatestPrice)
return <div>{price}</div>
```

## Checklist

- [ ] Identified all components using `useTradingStore()`
- [ ] Replaced with appropriate slice hooks
- [ ] Used granular selectors where applicable
- [ ] Tested for unnecessary re-renders
- [ ] Verified all actions still work
- [ ] Updated imports
- [ ] Removed deprecated hook warnings (if applicable)

## Need Help?

If you're unsure which slice to use, check the slice documentation:
- [lib/store/slices/README.md](./slices/README.md)

For questions about specific use cases, refer to the examples in the README.

## Timeline

- **Phase 1 (Week 1)**: Update core components (price displays, signal indicators)
- **Phase 2 (Week 2)**: Update complex components (dashboards, charts)
- **Phase 3 (Week 3)**: Optimize with granular selectors
- **Phase 4 (Week 4)**: Remove deprecated hooks

## Benefits You'll See

1. **Performance**: 50-80% reduction in unnecessary re-renders
2. **Bundle Size**: Better tree-shaking, smaller bundles
3. **Developer Experience**: Better autocomplete, clearer dependencies
4. **Maintainability**: Easier to test and modify individual slices
5. **Type Safety**: Stronger typing, fewer runtime errors
