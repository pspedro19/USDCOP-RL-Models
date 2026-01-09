# Financial Metrics Calculation System

A comprehensive real-time financial metrics calculation system for the USDCOP trading dashboard. This system provides institutional-grade performance analytics with automatic data fetching, caching, and real-time updates.

## Features

- **Comprehensive Metrics**: 30+ financial metrics including Sharpe ratio, Sortino ratio, Calmar ratio, VaR, CVaR, and more
- **Real-time Updates**: WebSocket support for live metric updates
- **Smart Caching**: Automatic caching to avoid redundant calculations
- **Equity Curves**: Build and analyze equity curves with drawdown analysis
- **Performance Analysis**: Advanced performance analytics and risk assessment
- **TypeScript**: Full type safety with comprehensive interfaces
- **React Hooks**: Easy-to-use React hooks for component integration

## Architecture

```
financial-metrics/
├── types.ts                 # Type definitions
├── MetricsCalculator.ts     # Core metric calculations
├── EquityCurveBuilder.ts    # Equity curve and drawdown analysis
├── PerformanceAnalyzer.ts   # Advanced performance analytics
├── index.ts                 # Exports
└── README.md               # This file
```

## Installation

The system is already integrated into the dashboard. No additional installation required.

## Usage

### Basic Usage with React Hook

```typescript
import { useFinancialMetrics } from '@/hooks/useFinancialMetrics';

function MyComponent() {
  const { metrics, isLoading, error } = useFinancialMetrics({
    initialCapital: 100000,
    pollInterval: 30000, // 30 seconds
  });

  if (isLoading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;

  return (
    <div>
      <p>Total P&L: {metrics.totalPnL}</p>
      <p>Sharpe Ratio: {metrics.sharpeRatio}</p>
      <p>Win Rate: {metrics.winRate * 100}%</p>
    </div>
  );
}
```

### Real-time Metrics

```typescript
import { useRealtimeMetrics } from '@/hooks/useFinancialMetrics';

function RealtimePanel() {
  const { metrics, lastUpdate } = useRealtimeMetrics({
    initialCapital: 100000,
  });

  return (
    <div>
      <p>Updated: {new Date(lastUpdate).toLocaleTimeString()}</p>
      <p>Current P&L: {metrics.totalPnL}</p>
    </div>
  );
}
```

### Manual Calculation

```typescript
import { MetricsCalculator } from '@/lib/services/financial-metrics';
import { Trade, Position } from '@/lib/services/financial-metrics/types';

const trades: Trade[] = [
  // Your trade data
];

const positions: Position[] = [
  // Your position data
];

const metrics = MetricsCalculator.calculateMetrics(trades, positions, {
  initialCapital: 100000,
  riskFreeRate: 0.03,
});

console.log('Sharpe Ratio:', metrics.sharpeRatio);
console.log('Max Drawdown:', metrics.maxDrawdownPercent);
```

### Equity Curve Analysis

```typescript
import { EquityCurveBuilder } from '@/lib/services/financial-metrics';

// Build equity curve
const equityCurve = EquityCurveBuilder.buildEquityCurve(
  trades,
  100000 // initial capital
);

// Get max drawdown
const maxDrawdown = EquityCurveBuilder.getMaxDrawdown(equityCurve);
console.log('Max Drawdown:', maxDrawdown.percent, '%');

// Get current drawdown
const currentDrawdown = EquityCurveBuilder.getCurrentDrawdown(equityCurve);
console.log('Current Drawdown:', currentDrawdown.percent, '%');
```

### Performance Analysis

```typescript
import { PerformanceAnalyzer } from '@/lib/services/financial-metrics';

// Generate comprehensive summary
const summary = PerformanceAnalyzer.generateSummary(trades, positions);

console.log('Risk Level:', summary.riskIndicators.riskLevel);
console.log('Warnings:', summary.riskIndicators.warnings);
console.log('Top Trades:', summary.topTrades);

// Compare time periods
const comparison = PerformanceAnalyzer.comparePeriods(
  trades,
  { start: lastMonth, end: thisMonth },
  { start: thisMonth, end: now }
);

console.log('Improvements:', comparison.improvements);
console.log('Deteriorations:', comparison.deteriorations);
```

## Metrics Reference

### P&L Metrics

- **totalPnL**: Total profit/loss (realized + unrealized)
- **realizedPnL**: Profit/loss from closed trades
- **unrealizedPnL**: Profit/loss from open positions
- **dailyPnL**: P&L in last 24 hours
- **weeklyPnL**: P&L in last 7 days
- **monthlyPnL**: P&L in last 30 days

### Return Metrics

- **totalReturn**: Overall return percentage
- **dailyReturn**: Return in last 24 hours
- **weeklyReturn**: Return in last 7 days
- **monthlyReturn**: Return in last 30 days
- **annualizedReturn**: Annualized return rate

### Performance Ratios

- **sharpeRatio**: Risk-adjusted return (higher is better, >1 is good, >2 is excellent)
- **sortinoRatio**: Downside risk-adjusted return (similar to Sharpe)
- **calmarRatio**: Return / Max drawdown (higher is better)
- **profitFactor**: Gross profit / Gross loss (>1 is profitable)

### Trade Statistics

- **totalTrades**: Total number of trades
- **winningTrades**: Number of profitable trades
- **losingTrades**: Number of losing trades
- **winRate**: Percentage of winning trades
- **avgWin**: Average winning trade amount
- **avgLoss**: Average losing trade amount
- **expectancy**: Expected value per trade
- **payoffRatio**: Average win / Average loss

### Risk Metrics

- **maxDrawdown**: Largest peak-to-trough decline (absolute)
- **maxDrawdownPercent**: Largest drawdown as percentage
- **currentDrawdown**: Current drawdown amount
- **valueAtRisk95**: 95% Value at Risk
- **expectedShortfall**: Conditional VaR (CVaR)
- **volatility**: Annualized standard deviation
- **downsideVolatility**: Downside standard deviation

### Position Metrics

- **openPositions**: Number of open positions
- **avgPositionSize**: Average position size
- **largestPosition**: Largest position size
- **exposure**: Total market exposure

## Configuration Options

```typescript
interface MetricsOptions {
  riskFreeRate?: number;        // Annual risk-free rate (default: 0.03)
  confidenceLevel?: number;     // For VaR/CVaR (default: 0.95)
  tradingDaysPerYear?: number;  // Default: 252
  initialCapital?: number;      // For equity curve calculations (default: 100000)
  includeOpenPositions?: boolean; // Include unrealized P&L (default: true)
}
```

## Data Sources

The system automatically fetches data from:

- `/api/trading/trades` - Historical and current trades
- `/api/trading/positions` - Open positions
- `ws://localhost:8000/ws/trades` - Real-time WebSocket updates

## Trade Data Format

```typescript
interface Trade {
  id: string;
  timestamp: number;
  symbol: string;
  side: 'buy' | 'sell' | 'long' | 'short';
  quantity: number;
  entryPrice: number;
  exitPrice?: number;
  entryTime: number;
  exitTime?: number;
  pnl: number;
  pnlPercent: number;
  commission: number;
  status: 'open' | 'closed';
  duration?: number; // in minutes
  strategy?: string;
}
```

## Position Data Format

```typescript
interface Position {
  id: string;
  symbol: string;
  side: 'long' | 'short';
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  unrealizedPnL: number;
  unrealizedPnLPercent: number;
  openTime: number;
  duration: number; // in minutes
  strategy?: string;
}
```

## Performance Considerations

- **Caching**: Calculations are cached for 5 seconds to avoid redundant processing
- **Polling**: Default poll interval is 30 seconds (configurable)
- **WebSocket**: Real-time updates via WebSocket for instant metric updates
- **Memoization**: React hooks use useMemo and useCallback for optimization

## Risk Assessment

The system automatically evaluates risk levels based on:

- Maximum drawdown > 20%
- Current drawdown > 15%
- Volatility > 30%
- Sharpe ratio < 0.5
- Win rate < 40%
- Profit factor < 1.2
- High market exposure

Risk levels: `low`, `medium`, `high`

## Examples

See `RealTimeMetricsPanel.tsx` for a complete implementation example.

## API Reference

### MetricsCalculator

- `calculateMetrics(trades, positions, options)`: Calculate all metrics
- `updateCalmarRatio(metrics)`: Update Calmar ratio after drawdown calculation

### EquityCurveBuilder

- `buildEquityCurve(trades, initialCapital, includeIntraday)`: Build equity curve
- `buildLiveEquityCurve(trades, positions, initialCapital)`: Include unrealized P&L
- `calculateDrawdowns(equityCurve)`: Find all drawdown periods
- `getMaxDrawdown(equityCurve)`: Get maximum drawdown info
- `getCurrentDrawdown(equityCurve)`: Get current drawdown
- `smoothCurve(curve, window)`: Smooth curve with moving average
- `resampleCurve(curve, intervalMs)`: Resample to fixed intervals
- `calculateRollingReturns(curve, windowMs)`: Calculate rolling returns
- `getUnderwaterPlot(curve)`: Get drawdown plot data

### PerformanceAnalyzer

- `generateSummary(trades, positions, options)`: Generate performance summary
- `analyzeRisk(metrics)`: Analyze risk levels and warnings
- `analyzeTimePeriod(trades, period, options)`: Analyze specific time period
- `groupByStrategy(trades, positions, options)`: Group metrics by strategy
- `comparePeriods(trades, period1, period2, options)`: Compare two periods
- `calculateRollingMetrics(trades, windowDays, options)`: Rolling metrics
- `calculateMonthlyReturns(trades, initialCapital)`: Monthly returns
- `calculateStreaks(trades)`: Win/loss streak analysis
- `calculateTradeDistribution(trades)`: Trade distribution analysis

## License

Part of the USDCOP Trading Dashboard project.
