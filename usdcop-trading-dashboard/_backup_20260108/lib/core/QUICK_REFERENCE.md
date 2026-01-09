# Quick Reference - Core Library

## Import Statements

```typescript
// Import everything
import * from '@/lib/core';

// Import specific items
import {
  // Interfaces
  IDataProvider,
  IWebSocketProvider,
  IRiskCalculator,
  ISubscribable,

  // Container
  ServiceContainer,
  ServiceLifetime,
  ServiceKeys,
  getGlobalContainer,
  resetGlobalContainer,
} from '@/lib/core';
```

## Service Container Cheat Sheet

### Registration

```typescript
const container = getGlobalContainer();

// Singleton (shared instance)
container.registerSingleton('key', () => new MyService());

// Transient (new instance each time)
container.registerTransient('key', () => new MyService());

// Instance (existing object)
container.registerInstance('key', myServiceInstance);

// Generic registration
container.register('key', () => new MyService(), ServiceLifetime.SINGLETON);
```

### Resolution

```typescript
// Get service (throws if not found)
const service = container.get<IMyService>('key');

// Try get (returns undefined if not found)
const service = container.tryGet<IMyService>('key');

// Check if service exists
if (container.has('key')) {
  // Service is registered
}
```

### Predefined Keys

```typescript
ServiceKeys.DATA_PROVIDER          // 'dataProvider'
ServiceKeys.WEBSOCKET_PROVIDER     // 'websocketProvider'
ServiceKeys.RISK_CALCULATOR        // 'riskCalculator'
ServiceKeys.LOGGER                 // 'logger'
ServiceKeys.HTTP_CLIENT            // 'httpClient'
ServiceKeys.CACHE                  // 'cache'
ServiceKeys.EVENT_BUS              // 'eventBus'
```

### Management

```typescript
// Get all registered keys
const keys = container.getKeys();

// Get registration info
const info = container.getRegistration('key');
// Returns: { lifetime: ServiceLifetime, hasInstance: boolean }

// Unregister a service
container.unregister('key');

// Clear all services
container.clear();

// Dispose all services (calls dispose() on services)
container.dispose();
```

### Child Containers

```typescript
const parent = getGlobalContainer();
const child = parent.createChild();

// Child can access parent services
const parentService = child.get(ServiceKeys.DATA_PROVIDER);

// Child can override parent services
child.registerInstance('key', childService);
```

## Interface Implementations

### IDataProvider

```typescript
class MyDataProvider implements IDataProvider {
  async getRealTimeData(symbol: string): Promise<MarketDataPoint> { }
  async getHistoricalData(symbol: string, timeframe: string): Promise<CandlestickExtended[]> { }
  async getSymbolStats(symbol: string): Promise<SymbolStats> { }
  async getBacktestResults(strategyCode: string): Promise<BacktestResult> { }
  async getPerformanceMetrics(): Promise<PerformanceMetrics> { }
  async healthCheck(): Promise<boolean> { }
}
```

### IWebSocketProvider

```typescript
class MyWSProvider implements IWebSocketProvider {
  async connect(): Promise<void> { }
  disconnect(): void { }
  subscribe(channel: string, handler: (data: unknown) => void): () => void { }
  unsubscribe(channel: string): void { }
  isConnected(): boolean { }
  getConnectionState(): ConnectionState { }
}
```

### IRiskCalculator

```typescript
class MyRiskCalculator implements IRiskCalculator {
  calculatePositionRisk(position: Position): RiskMetrics { }
  calculatePortfolioRisk(positions: Position[]): RiskMetrics { }
  calculatePositionSize(...): PositionSize { }
  calculateRiskLevels(...): RiskLevels { }
  validateTrade(...): { isValid: boolean; violations: string[] } { }
  calculateVaR(...): number { }
  calculateExpectedShortfall(...): number { }
}
```

### ISubscribable

```typescript
class MyObservable<T> implements ISubscribable<T> {
  subscribe(handler: (data: T) => void): () => void { }
  unsubscribe(handler: (data: T) => void): void { }
  unsubscribeAll(): void { }
  getSubscriberCount(): number { }
}
```

## React Integration

### Custom Hook Pattern

```typescript
import { useMemo } from 'react';
import { getGlobalContainer, ServiceKeys, IDataProvider } from '@/lib/core';

export function useDataProvider(): IDataProvider {
  return useMemo(() => {
    return getGlobalContainer().get<IDataProvider>(ServiceKeys.DATA_PROVIDER);
  }, []);
}
```

### Component Usage

```typescript
export function MyComponent() {
  const dataProvider = useDataProvider();

  useEffect(() => {
    const loadData = async () => {
      const data = await dataProvider.getRealTimeData('USDCOP');
      // Use data...
    };
    loadData();
  }, [dataProvider]);

  return <div>...</div>;
}
```

## Testing

### Setup Test Container

```typescript
import { resetGlobalContainer, getGlobalContainer } from '@/lib/core';

beforeEach(() => {
  resetGlobalContainer();
});

test('my test', () => {
  const container = getGlobalContainer();

  // Register mock
  const mock = { getData: jest.fn() };
  container.registerInstance(ServiceKeys.DATA_PROVIDER, mock);

  // Test code...
});
```

### Mock Services

```typescript
const mockDataProvider: IDataProvider = {
  getRealTimeData: jest.fn().mockResolvedValue({ price: 4000 }),
  getHistoricalData: jest.fn().mockResolvedValue([]),
  getSymbolStats: jest.fn().mockResolvedValue({}),
  getBacktestResults: jest.fn().mockResolvedValue({}),
  getPerformanceMetrics: jest.fn().mockResolvedValue({}),
  healthCheck: jest.fn().mockResolvedValue(true),
};

container.registerInstance(ServiceKeys.DATA_PROVIDER, mockDataProvider);
```

## Common Patterns

### Dependency Injection in Factory

```typescript
container.registerSingleton('complexService', () => {
  // Resolve dependencies
  const dataProvider = container.get<IDataProvider>(ServiceKeys.DATA_PROVIDER);
  const riskCalc = container.get<IRiskCalculator>(ServiceKeys.RISK_CALCULATOR);

  // Inject and return
  return new ComplexService(dataProvider, riskCalc);
});
```

### Service with Cleanup

```typescript
class MyService {
  private timer: NodeJS.Timeout;

  constructor() {
    this.timer = setInterval(() => { /* ... */ }, 1000);
  }

  // Cleanup method called by container.dispose()
  dispose() {
    clearInterval(this.timer);
  }
}
```

### Observable Pattern

```typescript
const observable = new SimpleObservable<number>();

// Subscribe
const unsubscribe = observable.subscribe((value) => {
  console.log('New value:', value);
});

// Emit
observable.emit(42);

// Unsubscribe
unsubscribe();
```

## Application Setup

### Entry Point (app.tsx or _app.tsx)

```typescript
import { setupServices } from '@/lib/setup-services';

// Initialize on app start
setupServices();

function App() {
  return <YourApp />;
}
```

### Setup File (lib/setup-services.ts)

```typescript
import { getGlobalContainer, ServiceKeys } from '@/lib/core';
import { MyDataProvider } from './providers/DataProvider';

export function setupServices() {
  const container = getGlobalContainer();

  container.registerSingleton(
    ServiceKeys.DATA_PROVIDER,
    () => new MyDataProvider({ baseUrl: process.env.API_URL })
  );

  // Register other services...

  return container;
}
```

## Best Practices

1. ✅ Register services at application startup
2. ✅ Use interfaces for type contracts
3. ✅ Use ServiceKeys constants
4. ✅ Create custom hooks for React
5. ✅ Implement dispose() for cleanup
6. ✅ Use singleton for stateful services
7. ✅ Use transient for utilities
8. ✅ Reset container in test setup
9. ✅ Program against interfaces, not implementations
10. ✅ Use TypeScript generics for type safety

## Common Mistakes

1. ❌ Forgetting to register services before use
2. ❌ Using string literals instead of ServiceKeys
3. ❌ Not disposing resources
4. ❌ Creating services inside components
5. ❌ Not resetting container between tests
6. ❌ Mixing service lifetimes incorrectly
7. ❌ Not using TypeScript generics
8. ❌ Circular dependencies in factories

## File Structure

```
lib/core/
├── interfaces/           # Interface abstractions
│   ├── IDataProvider.ts
│   ├── IWebSocketProvider.ts
│   ├── IRiskCalculator.ts
│   ├── ISubscribable.ts
│   ├── IExportHandler.ts
│   └── index.ts
├── container/           # DI container
│   ├── ServiceContainer.ts
│   └── index.ts
├── examples/            # Usage examples
│   └── basic-usage.ts
├── __tests__/          # Unit tests
│   └── ServiceContainer.test.ts
├── index.ts            # Main export
├── README.md           # Full documentation
└── QUICK_REFERENCE.md  # This file
```
