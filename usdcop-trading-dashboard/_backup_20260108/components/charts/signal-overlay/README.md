# Signal Overlay Component System

A professional trading signal overlay system for candlestick charts in the USDCOP trading dashboard. Built with lightweight-charts and React.

## Features

- **Signal Markers**: Visual indicators for BUY/SELL/HOLD signals with confidence levels
- **Stop Loss/Take Profit Lines**: Horizontal dashed lines showing SL/TP levels
- **Position Shading**: Colored areas between entry and exit showing P&L
- **Interactive Tooltips**: Detailed signal information on hover
- **Real-time Updates**: WebSocket integration for live signal updates
- **Advanced Filtering**: Filter by date, confidence, type, and status
- **Performance Optimized**: Virtualized rendering for many signals

## Installation

All files are already in place:

```
components/charts/
├── SignalOverlay.tsx          # Main component
└── signal-overlay/
    ├── types.ts               # Type definitions
    ├── SignalMarker.tsx       # Signal marker utilities
    ├── PositionShading.tsx    # Position area shading
    ├── StopLossTakeProfit.tsx # SL/TP line management
    └── index.ts               # Central exports

hooks/
└── useSignalOverlay.ts        # Signal fetching and management hook
```

## Quick Start

### Basic Usage

```tsx
import { useRef, useEffect } from 'react'
import { createChart, ISeriesApi } from 'lightweight-charts'
import SignalOverlay from '@/components/charts/SignalOverlay'
import { SignalData } from '@/components/charts/signal-overlay/types'

function TradingChart() {
  const chartRef = useRef(null)
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)

  // Sample signals
  const signals: SignalData[] = [
    {
      id: 'sig_001',
      timestamp: '2025-12-17T10:30:00Z',
      time: 1702814400 as Time,
      type: 'BUY',
      confidence: 87.5,
      price: 4285.50,
      stopLoss: 4270.00,
      takeProfit: 4320.00,
      reasoning: ['RSI oversold', 'MACD bullish crossover'],
      riskScore: 3.2,
      expectedReturn: 0.81,
      timeHorizon: '15-30 min',
      modelSource: 'L5_PPO_LSTM',
      latency: 42,
      status: 'active',
    },
  ]

  return (
    <div ref={chartRef} style={{ position: 'relative' }}>
      <SignalOverlay
        signals={signals}
        candleSeries={candleSeriesRef.current}
        chartContainer={chartRef.current}
        showStopLoss={true}
        showTakeProfit={true}
        showTooltips={true}
      />
    </div>
  )
}
```

### With Auto-Update

```tsx
<SignalOverlay
  signals={[]}
  candleSeries={candleSeriesRef.current}
  chartContainer={chartRef.current}
  autoUpdate={true}
  websocketUrl="ws://localhost:3001"
  filter={{
    minConfidence: 70,
    showHold: false,
    showActive: true,
  }}
/>
```

### With Filtering

```tsx
const [filter, setFilter] = useState({
  startDate: new Date('2025-12-17'),
  endDate: new Date(),
  actionTypes: [OrderType.BUY, OrderType.SELL],
  minConfidence: 80,
  showHold: false,
  showActive: true,
  showClosed: true,
})

<SignalOverlay
  signals={signals}
  candleSeries={candleSeriesRef.current}
  filter={filter}
  onSignalClick={(signal) => console.log('Clicked:', signal)}
  onSignalHover={(signal) => console.log('Hover:', signal)}
/>
```

## Integration with Existing Charts

### ChartWithPositions Integration

```tsx
import SignalOverlay from '@/components/charts/SignalOverlay'
import { useSignalOverlay } from '@/hooks/useSignalOverlay'

export default function ChartWithPositions({ symbol, timeframe }) {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)

  // Fetch signals
  const { signals, loading, stats } = useSignalOverlay({
    autoRefresh: true,
    refreshInterval: 30000,
    filter: { minConfidence: 70 },
  })

  return (
    <div className="relative">
      <div ref={chartContainerRef}>
        {/* Your existing chart setup */}
      </div>

      {/* Add signal overlay */}
      <SignalOverlay
        signals={signals}
        candleSeries={candleSeriesRef.current}
        chartContainer={chartContainerRef.current}
        showStopLoss={true}
        showTakeProfit={true}
        showPositionAreas={true}
      />
    </div>
  )
}
```

## API Reference

### SignalOverlay Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `signals` | `SignalData[]` | `[]` | Array of trading signals to display |
| `candleSeries` | `ISeriesApi<'Candlestick'>` | `null` | Candlestick series instance |
| `chartContainer` | `HTMLDivElement` | `null` | Chart container element |
| `filter` | `SignalFilterOptions` | `{}` | Filter options for signals |
| `onSignalClick` | `(signal) => void` | - | Callback when signal is clicked |
| `onSignalHover` | `(signal) => void` | - | Callback when signal is hovered |
| `showStopLoss` | `boolean` | `true` | Show stop loss lines |
| `showTakeProfit` | `boolean` | `true` | Show take profit lines |
| `showPositionAreas` | `boolean` | `true` | Show position shading |
| `showTooltips` | `boolean` | `true` | Show hover tooltips |
| `autoUpdate` | `boolean` | `false` | Enable auto-refresh |
| `websocketUrl` | `string` | - | WebSocket URL for real-time updates |

### SignalData Type

```typescript
interface SignalData {
  id: string
  timestamp: string
  time: Time
  type: OrderType // 'BUY' | 'SELL' | 'HOLD'
  confidence: number // 0-100
  price: number
  stopLoss?: number
  takeProfit?: number
  reasoning: string[]
  riskScore: number // 0-10
  expectedReturn: number
  timeHorizon: string
  modelSource: string
  latency: number
  exitPrice?: number
  exitTimestamp?: string
  exitTime?: Time
  pnl?: number
  status: 'active' | 'closed' | 'cancelled'
}
```

### Filter Options

```typescript
interface SignalFilterOptions {
  startDate?: Date
  endDate?: Date
  actionTypes?: OrderType[]
  minConfidence?: number
  maxConfidence?: number
  showHold?: boolean
  showActive?: boolean
  showClosed?: boolean
  modelSources?: string[]
}
```

## Customization

### Custom Marker Colors

Modify `SignalMarker.tsx`:

```typescript
export function getMarkerConfig(signal: SignalData): SignalMarkerConfig {
  switch (signal.type) {
    case OrderType.BUY:
      return {
        position: 'belowBar',
        color: '#00ff00', // Your custom color
        shape: 'arrowUp',
        text: `BUY ${signal.confidence.toFixed(0)}%`,
      }
    // ...
  }
}
```

### Custom Tooltip Content

Modify the `SignalTooltip` component in `SignalOverlay.tsx` to add your own fields.

### Custom Position Shading

Modify `PositionShading.tsx`:

```typescript
export function getShadingColor(pnl: number, opacity: number = 0.15): string {
  // Your custom logic
  return pnl > 0 ? `rgba(0, 255, 0, ${opacity})` : `rgba(255, 0, 0, ${opacity})`
}
```

## Performance Tips

1. **Limit Signal Count**: Display only recent signals (last 200)
2. **Use Filtering**: Filter out low-confidence signals
3. **Disable Features**: Turn off position shading if not needed
4. **WebSocket vs Polling**: Use WebSocket for real-time, polling for historical

```typescript
// Good: Limited recent signals
const recentSignals = signals.slice(-200)

// Good: Filter low confidence
filter={{ minConfidence: 70 }}

// Good: Disable unused features
showPositionAreas={false}
```

## Examples

### Example 1: High Confidence Only

```tsx
<SignalOverlay
  signals={signals}
  candleSeries={series}
  filter={{
    minConfidence: 85,
    showHold: false,
  }}
/>
```

### Example 2: Today's Active Signals

```tsx
<SignalOverlay
  signals={signals}
  candleSeries={series}
  filter={{
    startDate: new Date(new Date().setHours(0, 0, 0, 0)),
    showActive: true,
    showClosed: false,
  }}
/>
```

### Example 3: BUY Signals Only

```tsx
<SignalOverlay
  signals={signals}
  candleSeries={series}
  filter={{
    actionTypes: [OrderType.BUY],
    minConfidence: 70,
  }}
/>
```

## Troubleshooting

### Markers Not Showing

- Ensure `candleSeries` is not null
- Check that signal timestamps match candle data timeframe
- Verify signals have valid `time` property (UNIX timestamp)

### Price Lines Not Rendering

- Confirm `showStopLoss` and `showTakeProfit` are true
- Check that signals have `stopLoss` or `takeProfit` values
- Ensure signals have `status: 'active'`

### Tooltips Not Working

- Verify `chartContainer` ref is set correctly
- Ensure `showTooltips` is true
- Check browser console for errors

## Advanced Usage

### Custom Signal Source

```typescript
import { useSignalOverlay } from '@/hooks/useSignalOverlay'

// Create custom hook
function useCustomSignals() {
  const { signals, loading } = useSignalOverlay({
    autoRefresh: true,
  })

  // Add custom processing
  const processedSignals = signals.map(signal => ({
    ...signal,
    // Your custom logic
  }))

  return { signals: processedSignals, loading }
}
```

### Signal Analytics

```typescript
import { calculatePositionStats } from '@/components/charts/signal-overlay'

const stats = calculatePositionStats(positionAreas)
console.log(`Win Rate: ${stats.winRate}%`)
console.log(`Avg P&L: $${stats.avgPnl}`)
```

## License

Part of the USDCOP Trading Dashboard project.
