# Trading Store - File Structure

## Complete Directory Tree

```
lib/store/
│
├── Core Implementation Files
│   ├── trading-store.ts (8.2KB)          # Main store composition
│   │   ├── Composes 5 slices
│   │   ├── Exports slice hooks (useMarketDataSlice, etc.)
│   │   ├── Exports granular selectors
│   │   ├── Backwards compatible hooks (deprecated)
│   │   └── DevTools + Persist + SubscribeWithSelector middleware
│   │
│   ├── index.ts (160B)                   # Public API exports
│   │   └── Re-exports from trading-store.ts
│   │
│   └── market-store.ts (932B)            # Legacy store (untouched)
│
├── Slices Directory (slices/)
│   │
│   ├── index.ts (319B)                   # Slice exports
│   │   └── Re-exports all slice modules
│   │
│   ├── marketDataSlice.ts (2.4KB)        # Market Data Slice
│   │   ├── Interface: MarketDataSlice
│   │   ├── State: marketData, isConnected, lastUpdate
│   │   ├── Actions: updateMarketData, setConnectionStatus
│   │   ├── Computed: getLatestPrice
│   │   └── Selectors: selectMarketData, selectIsConnected, etc.
│   │
│   ├── signalsSlice.ts (2.4KB)           # Signals Slice
│   │   ├── Interface: SignalsSlice
│   │   ├── State: signals[], latestSignal, isLoading, error
│   │   ├── Actions: addSignal, setSignals, clearSignals, etc.
│   │   └── Selectors: selectSignals, selectLatestSignal, etc.
│   │
│   ├── positionsSlice.ts (3.3KB)         # Positions Slice
│   │   ├── Interface: PositionsSlice
│   │   ├── State: positions[], activePosition, totalPnl, totalUnrealizedPnl
│   │   ├── Actions: setPositions, updatePosition, closePosition, etc.
│   │   ├── Computed: getPositionSide, getTotalPnl
│   │   └── Selectors: selectPositions, selectActivePosition, etc.
│   │
│   ├── tradesSlice.ts (2.1KB)            # Trades Slice
│   │   ├── Interface: TradesSlice
│   │   ├── State: trades[], todayTrades[], pendingOrders
│   │   ├── Actions: addTrade, setTrades
│   │   └── Selectors: selectTrades, selectTodayTrades, etc.
│   │
│   ├── uiSlice.ts (3.1KB)                # UI Slice
│   │   ├── Interface: UISlice
│   │   ├── State: timeframe, theme, indicators, volume, chartHeight
│   │   ├── Actions: setTimeframe, toggleIndicator, setTheme, etc.
│   │   └── Selectors: selectTimeframe, selectTheme, etc.
│   │
│   └── README.md (9.5KB)                 # Detailed slice documentation
│       ├── Slice responsibilities
│       ├── Usage examples
│       ├── API reference
│       └── Best practices
│
└── Documentation Files
    │
    ├── README.md (11KB)                  # Main documentation
    │   ├── Quick start guide
    │   ├── Architecture overview
    │   ├── Available hooks
    │   ├── Usage examples
    │   ├── Performance benefits
    │   ├── Testing guide
    │   └── FAQ
    │
    ├── ARCHITECTURE.md (16KB)            # System architecture
    │   ├── System overview diagram
    │   ├── Slice responsibilities
    │   ├── Data flow diagrams
    │   ├── Component subscription matrix
    │   ├── Performance characteristics
    │   ├── Middleware stack
    │   ├── State persistence strategy
    │   ├── Type safety flow
    │   ├── Extension points
    │   ├── Testing strategy
    │   └── Best practices
    │
    ├── MIGRATION_GUIDE.md (11KB)         # Migration instructions
    │   ├── Before/after comparison
    │   ├── Step-by-step migration
    │   ├── Example conversions
    │   ├── Performance impact
    │   ├── Common pitfalls
    │   ├── Testing checklist
    │   └── Timeline suggestions
    │
    ├── QUICK_REFERENCE.md (9.6KB)        # Developer cheat sheet
    │   ├── Import cheat sheet
    │   ├── Common patterns (14 examples)
    │   ├── State structure reference
    │   ├── Action reference
    │   ├── Selector reference
    │   ├── Performance tips
    │   ├── Testing snippets
    │   ├── TypeScript types
    │   └── Common mistakes
    │
    ├── REFACTORING_SUMMARY.md (12KB)     # This refactoring summary
    │   ├── Overview and statistics
    │   ├── Key improvements
    │   ├── Architecture details
    │   ├── Usage examples
    │   ├── Migration path
    │   ├── Testing strategy
    │   ├── Benefits realized
    │   ├── Files created
    │   └── Success metrics
    │
    └── FILE_STRUCTURE.md                 # This file
        └── Complete directory tree with descriptions
```

## File Size Summary

```
Implementation Files (Core):
├── trading-store.ts       8.2 KB   Main store composition
├── index.ts               160 B    Public exports
└── market-store.ts        932 B    Legacy store (untouched)
                          ──────
                          9.3 KB   Total core

Slice Files:
├── index.ts               319 B    Slice exports
├── marketDataSlice.ts     2.4 KB   Market data state
├── signalsSlice.ts        2.4 KB   Signals state
├── positionsSlice.ts      3.3 KB   Positions state
├── tradesSlice.ts         2.1 KB   Trades state
├── uiSlice.ts             3.1 KB   UI state
└── README.md              9.5 KB   Slice docs
                          ──────
                          23.1 KB  Total slices

Documentation Files:
├── README.md              11.0 KB  Main docs
├── ARCHITECTURE.md        16.0 KB  System design
├── MIGRATION_GUIDE.md     11.0 KB  Upgrade guide
├── QUICK_REFERENCE.md      9.6 KB  Cheat sheet
├── REFACTORING_SUMMARY.md 12.0 KB  Summary
└── FILE_STRUCTURE.md       2.0 KB  This file
                          ──────
                          61.6 KB  Total docs

Grand Total:              94.0 KB  All files
```

## Code Line Count

```
Implementation (TypeScript):
├── slices/marketDataSlice.ts      98 lines
├── slices/signalsSlice.ts        105 lines
├── slices/positionsSlice.ts      125 lines
├── slices/tradesSlice.ts          89 lines
├── slices/uiSlice.ts             130 lines
├── slices/index.ts                13 lines
└── trading-store.ts              263 lines
                                  ─────────
                                  823 lines  Total code

Documentation (Markdown):
├── README.md                     ~400 lines
├── slices/README.md              ~350 lines
├── ARCHITECTURE.md               ~600 lines
├── MIGRATION_GUIDE.md            ~450 lines
├── QUICK_REFERENCE.md            ~400 lines
├── REFACTORING_SUMMARY.md        ~500 lines
└── FILE_STRUCTURE.md             ~200 lines
                                  ──────────
                                  2,900 lines  Total docs

Grand Total:                      3,723 lines  All files
```

## Import Paths

### For Components

```typescript
// Slice hooks (recommended)
import {
  useMarketDataSlice,
  useSignalsSlice,
  usePositionsSlice,
  useTradesSlice,
  useUISlice,
} from '@/lib/store/trading-store'

// Granular selectors (best performance)
import {
  useTradingStore,
  selectLatestPrice,
  selectIsConnected,
  selectLatestSignal,
  selectActivePosition,
  selectTotalPnl,
} from '@/lib/store/trading-store'

// Types
import type {
  MarketData,
  Timeframe,
  Theme,
  TradingStore,
} from '@/lib/store/trading-store'

// Full store (legacy, not recommended)
import { useTradingStore } from '@/lib/store/trading-store'
```

### For Testing

```typescript
// Individual slice creators
import {
  createMarketDataSlice,
  createSignalsSlice,
  createPositionsSlice,
  createTradesSlice,
  createUISlice,
} from '@/lib/store/slices'

// Slice interfaces
import type {
  MarketDataSlice,
  SignalsSlice,
  PositionsSlice,
  TradesSlice,
  UISlice,
} from '@/lib/store/slices'
```

## File Dependencies

```
Component
    ↓
    uses
    ↓
trading-store.ts (Public API)
    ↓
    imports
    ↓
slices/index.ts
    ↓
    imports
    ↓
[5 Individual Slice Files]
    ├── marketDataSlice.ts
    ├── signalsSlice.ts
    ├── positionsSlice.ts
    ├── tradesSlice.ts
    └── uiSlice.ts
        ↓
        imports
        ↓
    @/types/trading (external)
        ├── TradingSignal
        ├── Position
        ├── Trade
        ├── OrderSide
        └── OrderStatus
```

## Maintenance Guide

### Adding a New Slice

1. Create `slices/yourSlice.ts`
2. Export from `slices/index.ts`
3. Import in `trading-store.ts`
4. Compose in store creation
5. Create slice hook
6. Add selectors
7. Update documentation

### Modifying a Slice

1. Edit slice file in `slices/`
2. Update interface if needed
3. Add/modify actions
4. Update selectors
5. Update tests
6. Update documentation

### Removing a Slice (Advanced)

1. Remove from `trading-store.ts` composition
2. Remove from `slices/index.ts` exports
3. Delete slice file
4. Update TypeScript types
5. Remove deprecated hooks
6. Update documentation

## Quick Navigation

| Need | Go To |
|------|-------|
| Quick start | [README.md](./README.md) |
| Slice details | [slices/README.md](./slices/README.md) |
| System design | [ARCHITECTURE.md](./ARCHITECTURE.md) |
| Migration help | [MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md) |
| Code examples | [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) |
| Refactoring info | [REFACTORING_SUMMARY.md](./REFACTORING_SUMMARY.md) |
| This reference | [FILE_STRUCTURE.md](./FILE_STRUCTURE.md) |

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-12-17 | Initial monolithic store |
| 2.0.0 | 2024-12-17 | Refactored to slice-based architecture |

## Related Files

```
Outside lib/store/:
├── types/trading.ts           # TradingSignal, Position, Trade types
├── components/               # Components using the store
├── hooks/                    # Custom hooks wrapping store
└── tests/                    # Store tests
```

---

**Last Updated:** December 17, 2024
**Architecture:** Slice-based Zustand Store
**Pattern:** Interface Segregation Principle (ISP)
