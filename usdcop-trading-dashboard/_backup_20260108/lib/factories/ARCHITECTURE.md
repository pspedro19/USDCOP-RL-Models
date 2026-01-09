# Factory Pattern Architecture

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Application Layer                            │
│  (React Components, Pages, API Routes)                              │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       ServiceFactory                                 │
│  - Main orchestrator for all services                               │
│  - configure()                                                       │
│  - createDataProvider()                                             │
│  - createExportHandler()                                            │
└──────────────┬────────────────────────────┬─────────────────────────┘
               │                            │
               ▼                            ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│  DataProviderFactory     │  │    ExportFactory         │
│  - create()              │  │    - create()            │
│  - createWithFallback()  │  │    - export()            │
│  - createFromEnvironment │  │    - validateData()      │
└──────────┬───────────────┘  └──────────┬───────────────┘
           │                             │
           ▼                             ▼
┌──────────────────────────────────────────────────────────┐
│                  Core Interfaces                          │
│  - IDataProvider                                          │
│  - IExportHandler                                         │
│  - IWebSocketProvider                                     │
└──────────┬───────────────────────────────┬───────────────┘
           │                               │
           ▼                               ▼
┌─────────────────────┐         ┌─────────────────────────┐
│   Data Providers    │         │   Export Handlers       │
├─────────────────────┤         ├─────────────────────────┤
│ • ApiDataProvider   │         │ • CsvExportHandler      │
│ • WebSocketProvider │         │ • ExcelExportHandler    │
│ • MockDataProvider  │         │ • PdfExportHandler      │
└─────────────────────┘         └─────────────────────────┘
```

## Component Interaction Flow

### 1. Data Fetching Flow

```
User Action (Click "Refresh")
    │
    ▼
React Component
    │
    ▼
ServiceFactory.createDataProvider('api')
    │
    ▼
DataProviderFactory.create('api')
    │
    ▼
ApiDataProvider instance
    │
    ▼
getRealTimeData()
    │
    ▼
HTTP Request to Trading API
    │
    ▼
MarketDataPoint[]
    │
    ▼
Component State Update
    │
    ▼
UI Renders New Data
```

### 2. Export Flow

```
User Action (Click "Export CSV")
    │
    ▼
Export Button Component
    │
    ▼
ServiceFactory.export('csv', data, 'filename')
    │
    ▼
ExportFactory.create('csv')
    │
    ▼
CsvExportHandler instance
    │
    ▼
handler.export(data, 'filename')
    │
    ▼
Validate Data
    │
    ▼
Convert to CSV format
    │
    ▼
Trigger Browser Download
    │
    ▼
User receives file
```

### 3. Real-time Subscription Flow

```
Component Mount
    │
    ▼
ServiceFactory.createDataProvider('websocket')
    │
    ▼
WebSocketDataProvider instance
    │
    ▼
subscribeToRealTimeUpdates(callback)
    │
    ▼
WebSocket Connection Established
    │
    ▼
Price Updates Stream
    │
    ▼
Callback Invoked with MarketDataPoint
    │
    ▼
Component State Updated
    │
    ▼
UI Re-renders
    │
    │ (On Component Unmount)
    ▼
Unsubscribe Function Called
    │
    ▼
WebSocket Connection Closed
```

## Class Hierarchy

### Data Provider Hierarchy

```
IDataProvider (Interface)
    ├── ApiDataProvider
    │   └── Uses: REST API, polling
    │
    ├── WebSocketDataProvider
    │   ├── Uses: WebSocket for real-time
    │   └── Delegates to ApiDataProvider for historical data
    │
    └── MockDataProvider
        └── Uses: In-memory mock data generation
```

### Export Handler Hierarchy

```
IExportHandler (Interface)
    ├── CsvExportHandler
    │   └── Outputs: .csv files
    │
    ├── ExcelExportHandler
    │   └── Outputs: .xlsx files (currently .csv)
    │
    └── PdfExportHandler
        └── Outputs: .pdf files (currently .html)
```

## Factory Pattern Benefits

### 1. Separation of Concerns

```
┌─────────────────────┐
│  Business Logic     │  ← Doesn't care about implementation
│  (Components)       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Factory Layer      │  ← Handles creation logic
│  (Factories)        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Implementation     │  ← Specific implementations
│  (Providers)        │
└─────────────────────┘
```

### 2. Dependency Injection

```typescript
// Old way: Tight coupling
class Dashboard {
  private api = new ApiDataProvider(); // Hard-coded dependency
}

// New way: Dependency injection
class Dashboard {
  constructor(private provider: IDataProvider) {} // Injected dependency
}

// Usage
const dashboard = new Dashboard(
  ServiceFactory.createDataProvider('api')  // Production
);

// Testing
const testDashboard = new Dashboard(
  ServiceFactory.createDataProvider('mock') // Testing
);
```

### 3. Open/Closed Principle

```
┌────────────────────────────────────────┐
│         Add New Provider               │
│  (Open for Extension)                  │
│                                        │
│  1. Implement IDataProvider            │
│  2. Add to DataProviderFactory         │
│  3. No existing code changes needed    │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│      Existing Code Unchanged           │
│  (Closed for Modification)             │
│                                        │
│  • Existing providers work as-is       │
│  • Existing tests pass                 │
│  • No breaking changes                 │
└────────────────────────────────────────┘
```

## Design Patterns Used

### 1. Factory Pattern

**Purpose:** Create objects without specifying exact class

```typescript
// Factory creates the right type based on configuration
const provider = DataProviderFactory.create('api');
// or
const provider = DataProviderFactory.create('mock');
// or
const provider = DataProviderFactory.create('websocket');
```

### 2. Strategy Pattern

**Purpose:** Different implementations of same interface

```typescript
// All implement IDataProvider, different strategies
class ApiDataProvider implements IDataProvider { }
class WebSocketDataProvider implements IDataProvider { }
class MockDataProvider implements IDataProvider { }

// Can swap strategies at runtime
let provider: IDataProvider = DataProviderFactory.create('api');
// Later...
provider = DataProviderFactory.create('websocket');
```

### 3. Singleton Pattern (ServiceFactory)

**Purpose:** Single configuration point

```typescript
// Configure once
ServiceFactory.configure({ ... });

// Use everywhere
const provider = ServiceFactory.createDataProvider();
```

### 4. Observer Pattern (Subscriptions)

**Purpose:** Real-time updates

```typescript
provider.subscribeToRealTimeUpdates((data) => {
  // Observer gets notified of changes
  console.log('New data:', data);
});
```

## Configuration Management

### Environment-Based Configuration

```
Development Environment
    │
    ├── DataProvider: Mock
    ├── ExportFormat: CSV
    └── Logging: Verbose

Production Environment
    │
    ├── DataProvider: API/WebSocket
    ├── ExportFormat: PDF
    └── Logging: Error only

Test Environment
    │
    ├── DataProvider: Mock
    ├── ExportFormat: CSV
    └── Logging: Silent
```

### Configuration Flow

```
Environment Variables (.env)
    │
    ▼
ServiceFactory.configure()
    │
    ▼
Factory Configuration Object
    │
    ▼
Used by all create() methods
    │
    ▼
Provider/Handler instances
```

## Testing Architecture

### Test Pyramid

```
              ┌─────────┐
             │   E2E    │  ← Full app with mock providers
            │───────────│
           │  Integration │  ← Multiple components + factories
          │───────────────│
         │   Unit Tests     │  ← Individual factories
        └───────────────────┘
```

### Test Strategy

```typescript
// Unit Tests (Fast, Isolated)
describe('DataProviderFactory', () => {
  it('creates API provider', () => {
    const provider = DataProviderFactory.create('api');
    expect(provider).toBeInstanceOf(ApiDataProvider);
  });
});

// Integration Tests (Components + Factories)
describe('Dashboard Integration', () => {
  it('fetches and displays data', async () => {
    const provider = DataProviderFactory.create('mock');
    const data = await provider.getRealTimeData();
    expect(data).toBeDefined();
  });
});

// E2E Tests (Full Application)
describe('Export Flow', () => {
  it('exports dashboard data to CSV', async () => {
    // Uses mock provider
    // Clicks export button
    // Verifies file download
  });
});
```

## Performance Considerations

### Provider Selection

```
┌─────────────────────────────────────────────┐
│  Provider Performance Characteristics       │
├─────────────────┬───────────────────────────┤
│ ApiDataProvider │ • HTTP overhead           │
│                 │ • Polling latency (2s)    │
│                 │ • Good for occasional use │
├─────────────────┼───────────────────────────┤
│ WebSocket       │ • Real-time updates       │
│                 │ • Low latency (<100ms)    │
│                 │ • Good for live trading   │
├─────────────────┼───────────────────────────┤
│ MockProvider    │ • Zero latency            │
│                 │ • No network calls        │
│                 │ • Perfect for testing     │
└─────────────────┴───────────────────────────┘
```

### Memory Management

```typescript
// Good: Clean up subscriptions
useEffect(() => {
  const unsubscribe = provider.subscribeToRealTimeUpdates(callback);
  return unsubscribe; // Cleanup on unmount
}, []);

// Bad: Memory leak
useEffect(() => {
  provider.subscribeToRealTimeUpdates(callback);
  // No cleanup!
}, []);
```

## Security Considerations

### API Key Management

```typescript
// Good: Environment variables
const provider = DataProviderFactory.create('api', {
  apiBaseUrl: process.env.NEXT_PUBLIC_API_URL,
  apiKey: process.env.API_KEY // Server-side only
});

// Bad: Hardcoded
const provider = new ApiDataProvider({
  apiKey: 'abc123' // Never do this!
});
```

### Export Security

```typescript
// Validate data before export
if (!ExportFactory.validateData('csv', data)) {
  throw new Error('Invalid data - potential security risk');
}

// Sanitize filenames
const safeFilename = filename.replace(/[^a-z0-9_-]/gi, '_');
```

## Extensibility Guide

### Adding a New Data Provider

1. **Create Implementation**
```typescript
// lib/services/providers/RedisDataProvider.ts
export class RedisDataProvider implements IDataProvider {
  async getRealTimeData() { /* ... */ }
  // ... other methods
}
```

2. **Update Factory**
```typescript
// lib/factories/DataProviderFactory.ts
export type DataProviderType = 'api' | 'mock' | 'websocket' | 'redis';

static create(type: DataProviderType, config?: DataProviderConfig): IDataProvider {
  switch (type) {
    case 'redis':
      return new RedisDataProvider(config);
    // ... other cases
  }
}
```

3. **Done!** No other changes needed

### Adding a New Export Format

1. **Create Handler**
```typescript
// lib/services/export/JsonExportHandler.ts
export class JsonExportHandler implements IExportHandler {
  async export(data, filename) { /* ... */ }
  getExtension() { return 'json'; }
  getMimeType() { return 'application/json'; }
  validateData(data) { return true; }
}
```

2. **Update Factory**
```typescript
// lib/factories/ExportFactory.ts
export type ExportFormat = 'csv' | 'excel' | 'pdf' | 'json';

static create(format: ExportFormat): IExportHandler {
  switch (format) {
    case 'json':
      return new JsonExportHandler();
    // ... other cases
  }
}
```

3. **Done!** No other changes needed

## Summary

This architecture provides:
- ✅ Clean separation of concerns
- ✅ Easy testing with mocks
- ✅ Flexible configuration
- ✅ Type-safe implementations
- ✅ Extensible design
- ✅ Production-ready code

All following SOLID principles and industry best practices.
