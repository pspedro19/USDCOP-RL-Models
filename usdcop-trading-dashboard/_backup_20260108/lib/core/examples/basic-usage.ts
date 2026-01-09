/**
 * Basic Usage Examples
 * ====================
 *
 * Examples demonstrating how to use the core library and dependency injection.
 */

import {
  ServiceContainer,
  ServiceLifetime,
  ServiceKeys,
  getGlobalContainer,
  IDataProvider,
  IWebSocketProvider,
  IRiskCalculator,
  ISubscribable,
  DataProviderConfig,
} from '@/lib/core';

// ============================================================================
// Example 1: Creating and Registering Services
// ============================================================================

/**
 * Example implementation of IDataProvider
 */
class ApiDataProvider implements IDataProvider {
  constructor(private config: DataProviderConfig) {}

  async getRealTimeData(symbol: string) {
    const response = await fetch(`${this.config.baseUrl}/market/${symbol}`);
    return response.json();
  }

  async getHistoricalData(symbol: string, timeframe: string) {
    const response = await fetch(
      `${this.config.baseUrl}/candles/${symbol}?timeframe=${timeframe}`
    );
    return response.json();
  }

  async getSymbolStats(symbol: string) {
    const response = await fetch(`${this.config.baseUrl}/stats/${symbol}`);
    return response.json();
  }

  async getBacktestResults(strategyCode: string) {
    const response = await fetch(
      `${this.config.baseUrl}/backtest/${strategyCode}`
    );
    return response.json();
  }

  async getPerformanceMetrics() {
    const response = await fetch(`${this.config.baseUrl}/performance`);
    return response.json();
  }

  async healthCheck() {
    try {
      const response = await fetch(`${this.config.baseUrl}/health`);
      return response.ok;
    } catch {
      return false;
    }
  }
}

/**
 * Setup function to register all services
 */
export function setupServices() {
  const container = getGlobalContainer();

  // Register data provider as singleton
  container.registerSingleton(ServiceKeys.DATA_PROVIDER, () => {
    return new ApiDataProvider({
      baseUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001',
      timeout: 30000,
      cacheEnabled: true,
      cacheTTL: 60000,
    });
  });

  // Register other services...
  // container.registerSingleton(ServiceKeys.WEBSOCKET_PROVIDER, ...);
  // container.registerSingleton(ServiceKeys.RISK_CALCULATOR, ...);

  return container;
}

// ============================================================================
// Example 2: Using Services in Components (React)
// ============================================================================

/**
 * Custom hook to access data provider
 */
export function useDataProvider(): IDataProvider {
  const container = getGlobalContainer();
  return container.get<IDataProvider>(ServiceKeys.DATA_PROVIDER);
}

/**
 * Example React component using the hook
 */
export function MarketDataComponent() {
  const dataProvider = useDataProvider();

  const loadData = async () => {
    try {
      const data = await dataProvider.getRealTimeData('USDCOP');
      console.log('Market data:', data);
    } catch (error) {
      console.error('Failed to load market data:', error);
    }
  };

  // ... rest of component
  return null;
}

// ============================================================================
// Example 3: Manual Service Resolution
// ============================================================================

/**
 * Get a service manually
 */
export function manualServiceUsage() {
  const container = getGlobalContainer();

  // Get the data provider
  const dataProvider = container.get<IDataProvider>(ServiceKeys.DATA_PROVIDER);

  // Use it
  dataProvider
    .getSymbolStats('USDCOP')
    .then((stats) => {
      console.log('Symbol stats:', stats);
    });
}

// ============================================================================
// Example 4: Transient Services
// ============================================================================

/**
 * Example transient service (new instance each time)
 */
class RequestLogger {
  private requestId = Math.random().toString(36);

  log(message: string) {
    console.log(`[${this.requestId}] ${message}`);
  }
}

/**
 * Register transient service
 */
export function registerTransientServices() {
  const container = getGlobalContainer();

  container.registerTransient('requestLogger', () => new RequestLogger());
}

/**
 * Use transient service
 */
export function useTransientService() {
  const container = getGlobalContainer();

  // Each call gets a new instance
  const logger1 = container.get<RequestLogger>('requestLogger');
  const logger2 = container.get<RequestLogger>('requestLogger');

  logger1.log('First request'); // Different ID
  logger2.log('Second request'); // Different ID
}

// ============================================================================
// Example 5: Child Containers (Scoped Services)
// ============================================================================

/**
 * Create a child container for request-scoped services
 */
export function createRequestScope() {
  const globalContainer = getGlobalContainer();
  const requestContainer = globalContainer.createChild();

  // Register request-specific services
  requestContainer.registerInstance('requestId', crypto.randomUUID());
  requestContainer.registerInstance('userId', 'user-123');

  // Child can access parent services
  const dataProvider = requestContainer.get<IDataProvider>(
    ServiceKeys.DATA_PROVIDER
  );

  // And child services
  const requestId = requestContainer.get<string>('requestId');

  return { requestContainer, requestId };
}

// ============================================================================
// Example 6: Service with Dependencies
// ============================================================================

/**
 * Service that depends on other services
 */
class AnalyticsService {
  constructor(
    private dataProvider: IDataProvider,
    private riskCalculator: IRiskCalculator
  ) {}

  async analyzeSymbol(symbol: string) {
    const data = await this.dataProvider.getRealTimeData(symbol);
    // Use risk calculator for analysis
    // const risk = this.riskCalculator.calculatePositionRisk(...);
    return { data /* risk */ };
  }
}

/**
 * Register service with dependencies
 */
export function registerServiceWithDependencies() {
  const container = getGlobalContainer();

  container.registerSingleton('analyticsService', () => {
    // Resolve dependencies from container
    const dataProvider = container.get<IDataProvider>(ServiceKeys.DATA_PROVIDER);
    const riskCalculator = container.get<IRiskCalculator>(
      ServiceKeys.RISK_CALCULATOR
    );

    // Inject dependencies
    return new AnalyticsService(dataProvider, riskCalculator);
  });
}

// ============================================================================
// Example 7: Testing with Mocks
// ============================================================================

/**
 * Mock data provider for testing
 */
export class MockDataProvider implements IDataProvider {
  async getRealTimeData(symbol: string) {
    return {
      symbol,
      price: 4000,
      timestamp: Date.now(),
      volume: 1000000,
    };
  }

  async getHistoricalData() {
    return [];
  }

  async getSymbolStats() {
    return {
      symbol: 'USDCOP',
      price: 4000,
      open_24h: 3950,
      high_24h: 4050,
      low_24h: 3900,
      volume_24h: 10000000,
      change_24h: 50,
      change_percent_24h: 1.27,
      spread: 5,
      timestamp: new Date().toISOString(),
      source: 'mock',
    };
  }

  async getBacktestResults() {
    return {} as any;
  }

  async getPerformanceMetrics() {
    return {} as any;
  }

  async healthCheck() {
    return true;
  }
}

/**
 * Setup testing container with mocks
 */
export function setupTestingContainer() {
  const container = new ServiceContainer();

  container.registerInstance(ServiceKeys.DATA_PROVIDER, new MockDataProvider());

  return container;
}

// ============================================================================
// Example 8: Service Disposal
// ============================================================================

/**
 * Service with cleanup logic
 */
class WebSocketService implements IWebSocketProvider {
  private ws: WebSocket | null = null;

  async connect() {
    this.ws = new WebSocket('ws://localhost:8001');
  }

  disconnect() {
    this.ws?.close();
  }

  subscribe(channel: string, handler: (data: unknown) => void) {
    // Subscribe logic
    return () => {
      /* unsubscribe */
    };
  }

  unsubscribe(channel: string) {
    // Unsubscribe logic
  }

  isConnected() {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  getConnectionState() {
    return {
      state: 'connected' as const,
    };
  }

  // Cleanup method
  dispose() {
    this.disconnect();
    console.log('WebSocket service disposed');
  }
}

/**
 * Service will be automatically disposed when container is disposed
 */
export function demonstrateDisposal() {
  const container = new ServiceContainer();

  container.registerSingleton(
    ServiceKeys.WEBSOCKET_PROVIDER,
    () => new WebSocketService()
  );

  // Use the service
  const ws = container.get<IWebSocketProvider>(ServiceKeys.WEBSOCKET_PROVIDER);

  // Later, dispose all services
  container.dispose(); // Calls dispose() on WebSocketService
}

// ============================================================================
// Example 9: Observable Pattern
// ============================================================================

/**
 * Simple observable implementation
 */
class SimpleObservable<T> implements ISubscribable<T> {
  private handlers: Set<(data: T) => void> = new Set();

  subscribe(handler: (data: T) => void) {
    this.handlers.add(handler);
    return () => this.unsubscribe(handler);
  }

  unsubscribe(handler: (data: T) => void) {
    this.handlers.delete(handler);
  }

  unsubscribeAll() {
    this.handlers.clear();
  }

  getSubscriberCount() {
    return this.handlers.size;
  }

  // Emit value to all subscribers
  emit(value: T) {
    this.handlers.forEach((handler) => handler(value));
  }
}

/**
 * Use observable pattern
 */
export function demonstrateObservable() {
  const priceObservable = new SimpleObservable<number>();

  // Subscribe
  const unsubscribe = priceObservable.subscribe((price) => {
    console.log('Price updated:', price);
  });

  // Emit values
  priceObservable.emit(4000);
  priceObservable.emit(4010);

  // Unsubscribe
  unsubscribe();
}

// ============================================================================
// Example 10: Full Application Setup
// ============================================================================

/**
 * Complete application setup
 */
export function setupApplication() {
  console.log('Setting up application services...');

  const container = getGlobalContainer();

  // Register all services
  container.registerSingleton(ServiceKeys.DATA_PROVIDER, () => {
    return new ApiDataProvider({
      baseUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001',
      timeout: 30000,
      cacheEnabled: true,
      cacheTTL: 60000,
    });
  });

  // Add more service registrations here...

  console.log('Services registered:', container.getKeys());

  return container;
}
