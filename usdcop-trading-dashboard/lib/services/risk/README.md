# Risk Management Module

A modular risk management system following the Single Responsibility Principle (SRP). Each component has a single, well-defined responsibility.

## Architecture Overview

```
lib/services/risk/
├── types.ts                    # Type definitions
├── PortfolioTracker.ts         # Position tracking
├── RiskMetricsCalculator.ts    # Risk metric calculations
├── RiskAlertSystem.ts          # Alert generation and management
├── RealTimeRiskEngine.ts       # Orchestrator component
├── index.ts                    # Public API exports
└── README.md                   # This file
```

## Component Responsibilities

### 1. PortfolioTracker
**Single Responsibility:** Track and manage portfolio positions

**Capabilities:**
- Add/update positions
- Remove positions
- Retrieve position information
- Calculate portfolio summary statistics

**Example:**
```typescript
import { PortfolioTracker } from '@/lib/services/risk';

const tracker = new PortfolioTracker();
tracker.updatePosition({
  symbol: 'USDCOP',
  quantity: 1000,
  marketValue: 4200000,
  avgPrice: 4150,
  currentPrice: 4200,
  pnl: 50000,
  weight: 0.42,
  sector: 'FX',
  country: 'CO',
  currency: 'COP'
});

const positions = tracker.getAllPositions();
const summary = tracker.getPortfolioSummary();
```

### 2. RiskMetricsCalculator
**Single Responsibility:** Calculate risk metrics

**Capabilities:**
- Fetch risk metrics from Analytics API
- Calculate metrics from position data
- Compute VaR, leverage, volatility, etc.
- Map API responses to internal formats

**Example:**
```typescript
import { RiskMetricsCalculator } from '@/lib/services/risk';

const calculator = new RiskMetricsCalculator();

// Fetch from API
const metrics = await calculator.fetchRiskMetricsFromAPI('USDCOP', 10000000);

// Calculate from positions
const calculatedMetrics = calculator.calculateMetricsFromPositions(positions);
```

### 3. RiskAlertSystem
**Single Responsibility:** Manage risk alerts

**Capabilities:**
- Generate alerts based on threshold breaches
- Manage alert lifecycle (create, acknowledge, retrieve)
- Maintain alert history
- Provide alert statistics

**Example:**
```typescript
import { RiskAlertSystem } from '@/lib/services/risk';

const alertSystem = new RiskAlertSystem();

// Check thresholds and generate alerts
const alerts = alertSystem.checkRiskThresholds(metrics, thresholds);

// Get unacknowledged alerts
const unacknowledged = alertSystem.getAlerts(true);

// Acknowledge an alert
alertSystem.acknowledgeAlert('leverage-1234567890');

// Get statistics
const stats = alertSystem.getAlertStatistics();
```

### 4. RealTimeRiskEngine
**Single Responsibility:** Orchestrate risk management components

**Capabilities:**
- Coordinate PortfolioTracker, RiskMetricsCalculator, and RiskAlertSystem
- Manage real-time update lifecycle
- Implement subscription pattern for metrics updates
- Provide unified API for clients

**Example:**
```typescript
import { realTimeRiskEngine } from '@/lib/services/risk';

// Subscribe to updates
const callback = (metrics) => {
  console.log('New metrics:', metrics);
};
realTimeRiskEngine.subscribe(callback);

// Update position
realTimeRiskEngine.updatePosition(position);

// Get current metrics
const currentMetrics = realTimeRiskEngine.getRiskMetrics();

// Get alerts
const alerts = realTimeRiskEngine.getAlerts(true); // unacknowledged only

// Unsubscribe
realTimeRiskEngine.unsubscribe(callback);

// Cleanup
realTimeRiskEngine.destroy();
```

## Type Definitions

### Position
```typescript
interface Position {
  symbol: string;
  quantity: number;
  marketValue: number;
  avgPrice: number;
  currentPrice: number;
  pnl: number;
  weight: number;
  sector: string;
  country: string;
  currency: string;
}
```

### RiskAlert
```typescript
interface RiskAlert {
  id: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  type: string;
  message: string;
  timestamp: Date;
  acknowledged: boolean;
  position?: string;
  currentValue?: number;
  limitValue?: number;
  recommendation?: string;
  details?: Record<string, any>;
}
```

### RealTimeRiskMetrics
```typescript
interface RealTimeRiskMetrics {
  // Portfolio metrics
  portfolioValue: number;
  grossExposure: number;
  netExposure: number;
  leverage: number;

  // Risk measures
  portfolioVaR95: number;
  portfolioVaR99: number;
  expectedShortfall95: number;
  portfolioVolatility: number;

  // Drawdown metrics
  currentDrawdown: number;
  maximumDrawdown: number;

  // Liquidity metrics
  liquidityScore: number;
  timeToLiquidate: number;

  // Scenario analysis
  bestCaseScenario: number;
  worstCaseScenario: number;
  stressTestResults?: Record<string, number>;

  // Timestamps
  lastUpdated: Date;
  calculationTime: number;
}
```

### RiskThresholds
```typescript
interface RiskThresholds {
  maxLeverage: number;          // e.g., 5.0
  varLimit: number;             // e.g., 0.10 (10%)
  maxDrawdown: number;          // e.g., 0.15 (15%)
  minLiquidityScore: number;    // e.g., 0.7
  maxConcentration: number;     // e.g., 0.4 (40%)
}
```

## Configuration

Risk thresholds can be customized when creating a RealTimeRiskEngine instance:

```typescript
import { RealTimeRiskEngine } from '@/lib/services/risk';

const customThresholds = {
  maxLeverage: 3.0,
  varLimit: 0.08,
  maxDrawdown: 0.12,
  minLiquidityScore: 0.8,
  maxConcentration: 0.3
};

const engine = new RealTimeRiskEngine(customThresholds);
```

The singleton instance uses default thresholds:
```typescript
const DEFAULT_RISK_THRESHOLDS = {
  maxLeverage: 5.0,
  varLimit: 0.10,        // 10%
  maxDrawdown: 0.15,     // 15%
  minLiquidityScore: 0.7,
  maxConcentration: 0.4  // 40%
};
```

**Note:** A configuration service for loading thresholds from external config files is planned but not yet implemented.

## Subscription Pattern

The RealTimeRiskEngine implements the `ISubscribable<RealTimeRiskMetrics>` interface:

```typescript
interface ISubscribable<T> {
  subscribe(callback: (data: T) => void): void;
  unsubscribe(callback: (data: T) => void): void;
}
```

This allows components to subscribe to real-time metric updates:

```typescript
// Subscribe
const handleMetricsUpdate = (metrics: RealTimeRiskMetrics) => {
  console.log('Updated metrics:', metrics);
};
realTimeRiskEngine.subscribe(handleMetricsUpdate);

// Unsubscribe when done
realTimeRiskEngine.unsubscribe(handleMetricsUpdate);
```

## API Integration

The risk engine integrates with the Analytics API to fetch real risk metrics:

- **Endpoint:** `/api/analytics/risk-metrics`
- **Parameters:**
  - `symbol`: Trading symbol (e.g., 'USDCOP')
  - `portfolio_value`: Portfolio value in currency units
  - `days`: Historical data window (default: 30)

Example API call:
```
GET /api/analytics/risk-metrics?symbol=USDCOP&portfolio_value=10000000&days=30
```

## Logging

All components use the centralized logger from `@/lib/utils/logger`:

```typescript
import { createLogger } from '@/lib/utils/logger';
const logger = createLogger('ComponentName');

logger.debug('Debug message');
logger.info('Info message');
logger.warn('Warning message');
logger.error('Error message');
```

Each component creates a scoped logger with its class name for easy filtering and debugging.

## Testing

To test the risk module components:

```typescript
import {
  PortfolioTracker,
  RiskMetricsCalculator,
  RiskAlertSystem,
  RealTimeRiskEngine
} from '@/lib/services/risk';

// Test PortfolioTracker
const tracker = new PortfolioTracker();
tracker.updatePosition(mockPosition);
expect(tracker.getPositionCount()).toBe(1);

// Test RiskMetricsCalculator
const calculator = new RiskMetricsCalculator();
const metrics = calculator.calculateMetricsFromPositions([mockPosition]);
expect(metrics.portfolioValue).toBeGreaterThan(0);

// Test RiskAlertSystem
const alertSystem = new RiskAlertSystem();
const alerts = alertSystem.checkRiskThresholds(metrics, thresholds);
expect(alerts).toBeInstanceOf(Array);

// Test RealTimeRiskEngine (orchestrator)
const engine = new RealTimeRiskEngine();
engine.updatePosition(mockPosition);
const currentMetrics = engine.getRiskMetrics();
expect(currentMetrics).toBeDefined();
```

## Migration Guide

If you're migrating from the old monolithic `real-time-risk-engine.ts`:

1. **Update imports:**
   ```typescript
   // Old
   import { realTimeRiskEngine } from '@/lib/services/real-time-risk-engine';

   // New
   import { realTimeRiskEngine } from '@/lib/services/risk';
   ```

2. **API remains the same** - no code changes needed in your components

3. **Test thoroughly** after migration

4. The old file `real-time-risk-engine.ts` has been converted to a backwards-compatible wrapper that re-exports from the new structure

## Design Principles

This refactoring follows SOLID principles:

- **Single Responsibility Principle (SRP):** Each class has one reason to change
  - PortfolioTracker: Position data changes
  - RiskMetricsCalculator: Calculation logic changes
  - RiskAlertSystem: Alert rules changes
  - RealTimeRiskEngine: Orchestration logic changes

- **Open/Closed Principle:** Components are open for extension but closed for modification
  - New alert types can be added without modifying existing code
  - New metric calculations can be added to the calculator

- **Dependency Inversion:** High-level modules depend on abstractions
  - RealTimeRiskEngine depends on interfaces, not concrete implementations
  - ISubscribable interface defines the subscription contract

- **Interface Segregation:** Clients aren't forced to depend on interfaces they don't use
  - Each component exposes only the methods relevant to its responsibility

## Future Enhancements

- [ ] Configuration service for loading risk thresholds from external config
- [ ] WebSocket support for real-time metric streaming
- [ ] Historical metric storage and querying
- [ ] Advanced alert filtering and routing
- [ ] Metric snapshot/restore functionality
- [ ] Performance monitoring and optimization
- [ ] Unit and integration test suite
- [ ] Detailed API documentation with OpenAPI/Swagger

## License

Internal use only - USDCOP Trading Dashboard
