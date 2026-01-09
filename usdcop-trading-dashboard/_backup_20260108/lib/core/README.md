# Core Library

A lightweight dependency injection system with interface abstractions for the trading dashboard.

## Overview

This library provides:

1. **Interface Abstractions** - Type-safe interfaces for common services
2. **Service Container** - Simple dependency injection container using the registry pattern
3. **Global Container** - Singleton container instance for application-wide service management

## Directory Structure

```
lib/core/
├── interfaces/
│   ├── IDataProvider.ts          # Market data provider interface
│   ├── IWebSocketProvider.ts     # WebSocket connection interface
│   ├── IRiskCalculator.ts        # Risk calculation interface
│   ├── ISubscribable.ts          # Observable/subscription patterns
│   └── index.ts                  # Interface exports
├── container/
│   ├── ServiceContainer.ts       # DI container implementation
│   └── index.ts                  # Container exports
└── index.ts                      # Main exports
```

## Usage

### 1. Define Services

Implement the provided interfaces:

```typescript
import { IDataProvider } from '@/lib/core';

class MyDataProvider implements IDataProvider {
  async getRealTimeData(symbol: string) {
    // Implementation
  }

  async getHistoricalData(symbol: string, timeframe: string) {
    // Implementation
  }

  // ... other methods
}
```

### 2. Register Services

Register services in the container at application startup:

```typescript
import { getGlobalContainer, ServiceKeys } from '@/lib/core';

// Get the global container
const container = getGlobalContainer();

// Register a singleton service
container.registerSingleton(ServiceKeys.DATA_PROVIDER, () => new MyDataProvider());

// Register a transient service (new instance each time)
container.registerTransient('myTransientService', () => new MyTransientService());

// Register an existing instance
container.registerInstance('config', myConfigObject);
```

### 3. Consume Services

Retrieve and use services:

```typescript
import { getGlobalContainer, ServiceKeys, IDataProvider } from '@/lib/core';

function useMarketData() {
  const container = getGlobalContainer();
  const dataProvider = container.get<IDataProvider>(ServiceKeys.DATA_PROVIDER);

  // Use the data provider
  const data = await dataProvider.getRealTimeData('USDCOP');
}
```

### 4. React Hook Pattern

Create custom hooks for service consumption:

```typescript
import { useMemo } from 'react';
import { getGlobalContainer, ServiceKeys, IDataProvider } from '@/lib/core';

export function useDataProvider(): IDataProvider {
  return useMemo(() => {
    return getGlobalContainer().get<IDataProvider>(ServiceKeys.DATA_PROVIDER);
  }, []);
}
```

## Interfaces

### IDataProvider

Provides market data operations:

- `getRealTimeData(symbol)` - Get real-time market data
- `getHistoricalData(symbol, timeframe)` - Get historical candlestick data
- `getSymbolStats(symbol)` - Get 24h statistics
- `getBacktestResults(strategyCode)` - Get backtest results
- `getPerformanceMetrics()` - Get performance metrics

### IWebSocketProvider

Manages WebSocket connections:

- `connect()` - Connect to WebSocket server
- `disconnect()` - Disconnect from server
- `subscribe(channel, handler)` - Subscribe to a channel
- `unsubscribe(channel)` - Unsubscribe from a channel
- `isConnected()` - Check connection status

### IRiskCalculator

Calculates risk metrics:

- `calculatePositionRisk(position)` - Calculate position risk
- `calculatePortfolioRisk(positions)` - Calculate portfolio risk
- `calculatePositionSize(...)` - Calculate position sizing
- `calculateRiskLevels(...)` - Calculate stop loss/take profit
- `validateTrade(...)` - Validate trade against risk limits

### ISubscribable

Generic subscription pattern:

- `subscribe(handler)` - Subscribe to updates
- `unsubscribe(handler)` - Unsubscribe from updates
- `unsubscribeAll()` - Remove all subscriptions

## Service Lifetimes

The container supports three service lifetimes:

### Singleton (default)

A single instance is created and shared across all consumers:

```typescript
container.registerSingleton(ServiceKeys.DATA_PROVIDER, () => new MyDataProvider());
```

### Transient

A new instance is created each time the service is requested:

```typescript
container.registerTransient('myService', () => new MyService());
```

### Instance

Register an already-created instance:

```typescript
container.registerInstance('config', myConfigObject);
```

## Service Keys

Type-safe service keys are provided via `ServiceKeys`:

```typescript
export const ServiceKeys = {
  DATA_PROVIDER: 'dataProvider',
  WEBSOCKET_PROVIDER: 'websocketProvider',
  RISK_CALCULATOR: 'riskCalculator',
  LOGGER: 'logger',
  HTTP_CLIENT: 'httpClient',
  CACHE: 'cache',
  EVENT_BUS: 'eventBus',
} as const;
```

## Advanced Features

### Child Containers

Create scoped containers for isolation:

```typescript
const parentContainer = getGlobalContainer();
const childContainer = parentContainer.createChild();

// Register services in child
childContainer.register('scopedService', () => new ScopedService());

// Child can access parent services
const dataProvider = childContainer.get(ServiceKeys.DATA_PROVIDER);
```

### Service Disposal

Services implementing a `dispose()` method will be cleaned up:

```typescript
class MyService {
  dispose() {
    // Cleanup resources
  }
}

// When container is disposed, all services with dispose() are called
container.dispose();
```

### Type-Safe Service Resolution

Use TypeScript generics for type-safe service resolution:

```typescript
// Type is inferred
const dataProvider = container.get<IDataProvider>(ServiceKeys.DATA_PROVIDER);

// TypeScript knows the type
const data = await dataProvider.getRealTimeData('USDCOP');
```

## Best Practices

1. **Register at Startup** - Register all services during application initialization
2. **Use Interfaces** - Always program against interfaces, not implementations
3. **Singleton for Stateful** - Use singleton lifetime for stateful services (connections, caches)
4. **Transient for Stateless** - Use transient lifetime for stateless utilities
5. **Custom Hooks** - Create React hooks to consume services in components
6. **Type Safety** - Use `ServiceKeys` constants for type-safe service keys
7. **Dispose Resources** - Implement `dispose()` method for cleanup

## Example: Complete Setup

```typescript
// setup-container.ts
import { getGlobalContainer, ServiceKeys } from '@/lib/core';
import { DataProvider } from '@/lib/providers/DataProvider';
import { WebSocketProvider } from '@/lib/providers/WebSocketProvider';
import { RiskCalculator } from '@/lib/calculators/RiskCalculator';

export function setupContainer() {
  const container = getGlobalContainer();

  // Register data provider
  container.registerSingleton(
    ServiceKeys.DATA_PROVIDER,
    () => new DataProvider({ baseUrl: process.env.API_URL })
  );

  // Register WebSocket provider
  container.registerSingleton(
    ServiceKeys.WEBSOCKET_PROVIDER,
    () => new WebSocketProvider({ url: process.env.WS_URL })
  );

  // Register risk calculator
  container.registerSingleton(
    ServiceKeys.RISK_CALCULATOR,
    () => new RiskCalculator({ defaultRiskPercent: 2 })
  );

  return container;
}
```

```typescript
// app.tsx
import { setupContainer } from '@/lib/core/setup-container';

// Initialize container at app startup
setupContainer();

function App() {
  return <Dashboard />;
}
```

```typescript
// hooks/useDataProvider.ts
import { useMemo } from 'react';
import { getGlobalContainer, ServiceKeys, IDataProvider } from '@/lib/core';

export function useDataProvider(): IDataProvider {
  return useMemo(() => {
    return getGlobalContainer().get<IDataProvider>(ServiceKeys.DATA_PROVIDER);
  }, []);
}
```

```typescript
// components/MarketData.tsx
import { useDataProvider } from '@/hooks/useDataProvider';

export function MarketData() {
  const dataProvider = useDataProvider();

  const loadData = async () => {
    const data = await dataProvider.getRealTimeData('USDCOP');
    // Use data...
  };

  // ...
}
```

## Testing

Reset the container between tests:

```typescript
import { resetGlobalContainer } from '@/lib/core';

beforeEach(() => {
  resetGlobalContainer();
});
```

Mock services for testing:

```typescript
import { getGlobalContainer, ServiceKeys } from '@/lib/core';

const mockDataProvider = {
  getRealTimeData: jest.fn(),
  getHistoricalData: jest.fn(),
  // ...
};

getGlobalContainer().registerInstance(ServiceKeys.DATA_PROVIDER, mockDataProvider);
```
