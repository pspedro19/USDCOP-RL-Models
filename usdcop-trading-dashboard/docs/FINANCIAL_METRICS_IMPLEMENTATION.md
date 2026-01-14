# Financial Metrics Implementation Guide

## Overview

A complete, production-ready financial metrics calculation system for the USDCOP trading dashboard. This implementation provides institutional-grade performance analytics with real-time updates, smart caching, and comprehensive risk assessment.

## Files Created

### Core Services (`lib/services/financial-metrics/`)

1. **types.ts** (4.8 KB)
   - Complete TypeScript type definitions
   - Trade, Position, Signal interfaces
   - FinancialMetrics interface with 30+ metrics
   - Options and configuration types

2. **MetricsCalculator.ts** (15.9 KB)
   - Core metrics calculation engine
   - P&L calculations
   - Performance ratios (Sharpe, Sortino, Calmar)
   - Trade statistics
   - Risk metrics (VaR, CVaR, volatility)
   - Position metrics
   - Statistical helpers

3. **EquityCurveBuilder.ts** (11.3 KB)
   - Equity curve generation
   - Drawdown analysis
   - Peak-to-trough calculations
   - Curve smoothing and resampling
   - Rolling returns
   - Underwater plots

4. **PerformanceAnalyzer.ts** (12.5 KB)
   - Advanced performance analytics
   - Risk assessment and warnings
   - Time period comparisons
   - Strategy grouping
   - Rolling metrics
   - Monthly returns
   - Win/loss streak analysis
   - Trade distribution analysis

5. **index.ts** (461 B)
   - Central exports
   - Re-exports for convenience

6. **README.md** (Documentation)
   - Complete API reference
   - Usage examples
   - Configuration guide
   - Metrics reference

### React Integration (`hooks/`)

7. **useFinancialMetrics.ts** (11.2 KB)
   - React hook for metrics
   - Automatic data fetching from API
   - Smart caching (5-second cache)
   - Polling support (default: 30s)
   - WebSocket real-time updates
   - Error handling
   - Loading states
   - Manual refresh capability
   - Multiple hook variants:
     - `useFinancialMetrics()` - Standard
     - `useRealtimeMetrics()` - 5s polling + WebSocket
     - `useCachedMetrics()` - 1min polling, no WebSocket

### UI Components (`components/metrics/`)

8. **RealTimeMetricsPanel.tsx** (16.4 KB)
   - Complete metrics display component
   - 8 primary metric cards
   - Advanced metrics sections
   - Risk analysis panel
   - Trade statistics panel
   - Risk warnings display
   - Auto-refresh with loading states
   - Error handling with retry
   - Responsive grid layout
   - Color-coded indicators

9. **FinancialMetricsExample.tsx** (Usage examples)
   - 5 complete usage examples
   - BasicMetricsExample
   - RealtimeMetricsExample
   - PerformanceSummaryExample
   - EquityCurveExample
   - CompleteDashboardExample

## Quick Start

### 1. Using the Complete Panel

```tsx
import RealTimeMetricsPanel from '@/components/metrics/RealTimeMetricsPanel';

export default function TradingPage() {
  return (
    <div>
      <RealTimeMetricsPanel
        initialCapital={100000}
        pollInterval={30000}
        showAdvanced={true}
      />
    </div>
  );
}
```

### 2. Using the Hook Directly

```tsx
import { useFinancialMetrics } from '@/hooks/useFinancialMetrics';

export default function CustomMetrics() {
  const { metrics, isLoading, error, refresh } = useFinancialMetrics({
    initialCapital: 100000,
    pollInterval: 30000,
  });

  if (isLoading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;

  return (
    <div>
      <p>Total P&L: ${metrics.totalPnL.toFixed(2)}</p>
      <p>Win Rate: {(metrics.winRate * 100).toFixed(1)}%</p>
      <p>Sharpe Ratio: {metrics.sharpeRatio.toFixed(2)}</p>
      <button onClick={refresh}>Refresh</button>
    </div>
  );
}
```

### 3. Manual Calculation (without React)

```typescript
import { MetricsCalculator } from '@/lib/services/financial-metrics';

const trades = [
  // Your trade data
];

const metrics = MetricsCalculator.calculateMetrics(trades, [], {
  initialCapital: 100000,
  riskFreeRate: 0.03,
  tradingDaysPerYear: 252,
});

console.log('Metrics:', metrics);
```

## Metrics Calculated

### P&L Metrics (7)
- Total P&L (realized + unrealized)
- Realized P&L
- Unrealized P&L
- Daily, Weekly, Monthly P&L
- Total Return %

### Performance Ratios (4)
- Sharpe Ratio (risk-adjusted return)
- Sortino Ratio (downside risk-adjusted)
- Calmar Ratio (return / max drawdown)
- Profit Factor (gross profit / gross loss)

### Trade Statistics (10)
- Total Trades
- Winning/Losing Trades
- Win Rate %
- Average Win/Loss
- Largest Win/Loss
- Average Trade Duration
- Expectancy
- Payoff Ratio

### Risk Metrics (8)
- Maximum Drawdown ($ and %)
- Current Drawdown ($ and %)
- Value at Risk (95% VaR)
- Expected Shortfall (CVaR)
- Volatility (annualized)
- Downside Volatility

### Position Metrics (4)
- Open Positions Count
- Average Position Size
- Largest Position
- Total Exposure

### Additional (5)
- Equity Curve (array of points)
- Drawdown Curve (array of periods)
- Trading Days
- Information Ratio
- Kelly Fraction

**Total: 38+ metrics calculated**

## Data Flow

```
Trading API Backend (Port 8000)
         ↓
    /api/trading/trades      ← Fetches trade history
    /api/trading/positions   ← Fetches open positions
         ↓
  useFinancialMetrics Hook   ← Polls every 30s (configurable)
         ↓
  Data Transformation        ← Converts API format to Trade/Position types
         ↓
  Smart Cache (5s TTL)       ← Avoids redundant calculations
         ↓
  MetricsCalculator          ← Calculates all metrics
         ↓
  EquityCurveBuilder         ← Builds equity curve & drawdowns
         ↓
  PerformanceAnalyzer        ← Generates summary & risk analysis
         ↓
  React Components           ← Display metrics
         ↓
  WebSocket (Port 8000)      ← Real-time updates
         ↓
  Auto-refresh               ← Triggers recalculation
```

## API Endpoints Required

Your backend should provide these endpoints:

### 1. GET /api/trading/trades
Returns array of trades:
```json
{
  "trades": [
    {
      "id": "trade_123",
      "timestamp": "2024-01-15T10:30:00Z",
      "symbol": "USDCOP",
      "side": "buy",
      "quantity": 1000,
      "entry_price": 4250.5,
      "exit_price": 4255.0,
      "entry_time": "2024-01-15T10:30:00Z",
      "exit_time": "2024-01-15T11:45:00Z",
      "pnl": 4500,
      "pnl_percent": 0.11,
      "commission": 50,
      "status": "closed",
      "duration_minutes": 75,
      "strategy": "momentum"
    }
  ]
}
```

### 2. GET /api/trading/positions
Returns array of open positions:
```json
{
  "positions": [
    {
      "id": "pos_456",
      "symbol": "USDCOP",
      "side": "long",
      "quantity": 500,
      "entry_price": 4252.0,
      "current_price": 4256.5,
      "unrealized_pnl": 2250,
      "unrealized_pnl_percent": 0.11,
      "open_time": "2024-01-15T14:20:00Z",
      "strategy": "mean_reversion"
    }
  ]
}
```

### 3. WebSocket ws://localhost:8000/ws/trades (Optional)
Real-time updates:
```json
{
  "type": "trade_update",
  "data": {
    // Trade object
  }
}
```

## Configuration

### MetricsOptions

```typescript
{
  riskFreeRate: 0.03,           // 3% annual risk-free rate
  confidenceLevel: 0.95,        // 95% confidence for VaR/CVaR
  tradingDaysPerYear: 252,      // Standard trading days
  initialCapital: 100000,       // Starting capital
  includeOpenPositions: true    // Include unrealized P&L
}
```

### Hook Options

```typescript
{
  pollInterval: 30000,          // Polling interval in ms (30s)
  enableWebSocket: true,        // Enable real-time WebSocket
  autoRefresh: true,            // Auto-refresh on interval
  ...MetricsOptions             // All metric options
}
```

## Risk Assessment

The system automatically evaluates trading risk:

### Risk Factors Evaluated
- Maximum drawdown > 20% → High risk
- Current drawdown > 15% → Warning
- Volatility > 30% → High volatility
- Sharpe ratio < 0.5 → Poor risk-adjusted return
- Win rate < 40% → Low win rate
- Profit factor < 1.2 → Marginal profitability
- High market exposure → Overleveraged

### Risk Levels
- **Low**: Score 0-3 (Green)
- **Medium**: Score 4-6 (Yellow)
- **High**: Score 7+ (Red)

## Performance Optimizations

1. **Smart Caching**: 5-second cache prevents redundant calculations
2. **Memoization**: React hooks use useMemo/useCallback
3. **Parallel Fetching**: Trades and positions fetched simultaneously
4. **WebSocket**: Instant updates without polling overhead
5. **Conditional Rendering**: Loading states prevent unnecessary renders
6. **Error Boundaries**: Graceful error handling

## Testing

Test with mock data:

```typescript
const mockTrades = [
  {
    id: '1',
    timestamp: Date.now() - 3600000,
    symbol: 'USDCOP',
    side: 'buy' as const,
    quantity: 1000,
    entryPrice: 4250,
    exitPrice: 4260,
    entryTime: Date.now() - 3600000,
    exitTime: Date.now() - 1800000,
    pnl: 10000,
    pnlPercent: 0.23,
    commission: 100,
    status: 'closed' as const,
    duration: 30,
  },
  // Add more trades...
];

const metrics = MetricsCalculator.calculateMetrics(mockTrades, [], {
  initialCapital: 100000,
});

console.log('Test Metrics:', metrics);
```

## Integration with Existing Dashboard

### Option 1: Replace existing metrics component

```tsx
// In your trading page
import RealTimeMetricsPanel from '@/components/metrics/RealTimeMetricsPanel';

// Replace old component with:
<RealTimeMetricsPanel
  initialCapital={100000}
  pollInterval={30000}
  showAdvanced={true}
/>
```

### Option 2: Use alongside existing components

```tsx
import { useFinancialMetrics } from '@/hooks/useFinancialMetrics';
import YourExistingComponent from './YourExistingComponent';

export default function TradingPage() {
  const { metrics } = useFinancialMetrics();

  return (
    <div>
      <YourExistingComponent />
      <RealTimeMetricsPanel />
      {/* Or pass metrics to your components */}
      <CustomDisplay sharpeRatio={metrics?.sharpeRatio} />
    </div>
  );
}
```

## Troubleshooting

### No data showing
1. Check backend API is running on port 8000
2. Verify `/api/trading/trades` returns data
3. Check browser console for errors
4. Verify data format matches expected structure

### Slow performance
1. Increase cache duration in useFinancialMetrics
2. Reduce polling interval
3. Disable WebSocket if not needed
4. Filter trades to recent period only

### WebSocket not connecting
1. Verify backend WebSocket endpoint exists
2. Check browser console for WebSocket errors
3. Disable WebSocket: `enableWebSocket: false`
4. Rely on polling only

### Incorrect calculations
1. Verify initialCapital is correct
2. Check trade P&L values are accurate
3. Ensure exitTime is set for closed trades
4. Verify trade status ('open' vs 'closed')

## Future Enhancements

Possible additions:
- Chart visualizations for equity curve
- Drawdown periods chart
- Monthly returns heatmap
- Trade distribution histograms
- Strategy comparison charts
- Export to CSV/Excel
- PDF report generation
- Email alerts for risk thresholds
- Historical metric snapshots
- Backtesting integration

## Support

For questions or issues:
1. Check the README.md in `lib/services/financial-metrics/`
2. Review example code in `FinancialMetricsExample.tsx`
3. Inspect browser console for errors
4. Verify backend API responses

## Summary

This implementation provides:
- ✅ 38+ financial metrics
- ✅ Real-time updates via WebSocket
- ✅ Smart caching (5s TTL)
- ✅ Automatic data fetching
- ✅ Complete TypeScript types
- ✅ Production-ready components
- ✅ Comprehensive documentation
- ✅ Usage examples
- ✅ Error handling
- ✅ Risk assessment
- ✅ Performance optimized

Ready for immediate integration into your USDCOP trading dashboard!
