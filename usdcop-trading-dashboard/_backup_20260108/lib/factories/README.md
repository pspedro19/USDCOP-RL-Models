# Factory Pattern Implementation

This directory contains factory classes for service instantiation in the USD/COP Trading Dashboard.

## Overview

The factory pattern enables:
- **Easy Testing**: Swap real implementations with mock providers
- **Dependency Injection**: Configure services at runtime
- **Open/Closed Principle**: Add new implementations without modifying existing code
- **Single Responsibility**: Each factory handles one type of service creation

## Directory Structure

```
lib/factories/
├── DataProviderFactory.ts    # Creates data providers (API, Mock, WebSocket)
├── ExportFactory.ts           # Creates export handlers (CSV, Excel, PDF)
├── ServiceFactory.ts          # Main factory for all services
├── index.ts                   # Central export point
└── README.md                  # This file
```

## Factories

### 1. DataProviderFactory

Creates data providers for fetching market data.

**Available Types:**
- `api` - Real data from Trading API
- `mock` - Mock data for testing
- `websocket` - Real-time WebSocket data

**Usage:**

```typescript
import { DataProviderFactory } from '@/lib/factories';

// Create an API data provider
const apiProvider = DataProviderFactory.create('api');

// Create with configuration
const wsProvider = DataProviderFactory.create('websocket', {
  wsUrl: 'ws://localhost:8082/ws',
  apiBaseUrl: 'http://localhost:8000/api'
});

// Create with automatic fallback
const provider = DataProviderFactory.createWithFallback('api');

// Create based on environment
const envProvider = DataProviderFactory.createFromEnvironment();
```

### 2. ExportFactory

Creates export handlers for different file formats.

**Available Formats:**
- `csv` - CSV file export
- `excel` - Excel file export
- `pdf` - PDF report export

**Usage:**

```typescript
import { ExportFactory } from '@/lib/factories';

// Create a CSV export handler
const csvHandler = ExportFactory.create('csv');
await csvHandler.export(data, 'trading-report');

// Quick export
await ExportFactory.export('pdf', reportData, 'backtest-results', {
  title: 'Backtest Analysis Report',
  subtitle: 'Trading Strategy Performance',
  watermark: 'CONFIDENTIAL'
});

// Validate data before export
if (ExportFactory.validateData('csv', myData)) {
  await ExportFactory.export('csv', myData, 'data-export');
}
```

### 3. ServiceFactory

Main factory that orchestrates all service creation.

**Usage:**

```typescript
import { ServiceFactory } from '@/lib/factories';

// Configure factory defaults
ServiceFactory.configure({
  dataProvider: { type: 'api' },
  exportFormat: 'csv'
});

// Create services using configured defaults
const dataProvider = ServiceFactory.createDataProvider();
const exportHandler = ServiceFactory.createExportHandler();

// Override defaults
const mockProvider = ServiceFactory.createDataProvider('mock');
const pdfHandler = ServiceFactory.createExportHandler('pdf');

// Quick export
await ServiceFactory.export('excel', data, 'report');

// Get available options
const providers = ServiceFactory.getAvailableDataProviders();
// ['api', 'mock', 'websocket']

const formats = ServiceFactory.getAvailableExportFormats();
// ['csv', 'excel', 'pdf']
```

## Implementation Details

### Core Interfaces

All implementations follow these interfaces:

```typescript
// IDataProvider - lib/core/interfaces/IDataProvider.ts
interface IDataProvider {
  getRealTimeData(): Promise<MarketDataPoint[]>;
  getCandlestickData(...): Promise<CandlestickResponse>;
  getSymbolStats(symbol: string): Promise<SymbolStats>;
  checkHealth(): Promise<{ status: string; message?: string }>;
  subscribeToRealTimeUpdates(callback): () => void;
}

// IExportHandler - lib/core/interfaces/IExportHandler.ts
interface IExportHandler {
  export(data: any, filename: string, config?: ReportConfig): Promise<void>;
  getExtension(): string;
  getMimeType(): string;
  validateData(data: any): boolean;
}
```

### Data Provider Implementations

**ApiDataProvider** (`lib/services/providers/ApiDataProvider.ts`)
- Connects to Trading API REST endpoints
- Polls for real-time updates
- Automatic fallback to historical data when market is closed

**WebSocketDataProvider** (`lib/services/providers/WebSocketDataProvider.ts`)
- Real-time WebSocket streaming
- Automatic reconnection
- Falls back to API provider for historical data

**MockDataProvider** (`lib/services/mock/MockDataProvider.ts`)
- Generates realistic mock data
- Perfect for testing and development
- Simulates real-time updates

### Export Handler Implementations

**CsvExportHandler** (`lib/services/export/CsvExportHandler.ts`)
- Simple CSV file generation
- Handles special characters and commas
- Client-side file download

**ExcelExportHandler** (`lib/services/export/ExcelExportHandler.ts`)
- Excel file export (currently outputs CSV)
- Multi-sheet support (placeholder)
- TODO: Integrate xlsx library for proper Excel files

**PdfExportHandler** (`lib/services/export/PdfExportHandler.ts`)
- PDF report generation (currently outputs HTML)
- Support for metrics, tables, and charts
- Professional report formatting
- TODO: Integrate jsPDF or pdfmake for proper PDFs

## Testing with Mock Providers

The mock providers make testing easy:

```typescript
import { DataProviderFactory } from '@/lib/factories';

describe('Trading Dashboard', () => {
  it('should fetch market data', async () => {
    // Use mock provider for testing
    const provider = DataProviderFactory.create('mock');

    const data = await provider.getRealTimeData();

    expect(data).toHaveLength(1);
    expect(data[0].symbol).toBe('USDCOP');
  });
});
```

## Environment-Based Configuration

```typescript
// .env.local
NEXT_PUBLIC_DATA_PROVIDER=api
NEXT_PUBLIC_WS_URL=ws://localhost:8082/ws

// In your app
const provider = ServiceFactory.createDataProviderFromEnvironment();
// Automatically selects the right provider based on environment
```

## Advanced Usage

### Custom Configuration

```typescript
// app/config/services.ts
import { ServiceFactory } from '@/lib/factories';

export function initializeServices() {
  ServiceFactory.configure({
    dataProvider: {
      type: process.env.NODE_ENV === 'test' ? 'mock' : 'api',
      config: {
        apiBaseUrl: process.env.NEXT_PUBLIC_API_URL,
        timeout: 10000
      }
    },
    exportFormat: 'csv'
  });
}
```

### Dependency Injection

```typescript
// components/TradingDashboard.tsx
import { ServiceFactory } from '@/lib/factories';

export default function TradingDashboard() {
  const [provider] = useState(() =>
    ServiceFactory.createDataProviderWithFallback('api')
  );

  useEffect(() => {
    const unsubscribe = provider.subscribeToRealTimeUpdates((data) => {
      console.log('New market data:', data);
    });

    return unsubscribe;
  }, [provider]);

  // ...
}
```

### Export with Configuration

```typescript
import { ServiceFactory } from '@/lib/factories';

async function exportBacktestReport(results: BacktestResults) {
  const reportData = {
    metrics: [
      { label: 'Total Return', value: '15.3%' },
      { label: 'Sharpe Ratio', value: '1.82' },
      { label: 'Max Drawdown', value: '-8.4%' }
    ],
    tables: [
      {
        title: 'Trade Summary',
        headers: ['Date', 'Side', 'Price', 'P&L'],
        rows: results.trades.map(t => [
          t.date, t.side, t.price.toString(), t.pnl.toString()
        ])
      }
    ]
  };

  await ServiceFactory.export('pdf', reportData, 'backtest-report', {
    title: 'Backtest Analysis Report',
    subtitle: 'USD/COP Trading Strategy',
    author: 'Trading System',
    company: 'Professional Trading Platform',
    watermark: 'CONFIDENTIAL'
  });
}
```

## Benefits

1. **Testability**: Easy to swap implementations for testing
2. **Flexibility**: Change providers without changing application code
3. **Maintainability**: Single point of configuration
4. **Extensibility**: Add new providers without modifying existing code
5. **Type Safety**: Full TypeScript support with interfaces

## Future Enhancements

- [ ] Add Redis cache provider
- [ ] Implement proper Excel export with xlsx library
- [ ] Implement proper PDF export with jsPDF
- [ ] Add database data provider
- [ ] Add JSON export format
- [ ] Add XML export format
- [ ] Implement provider pooling for performance
- [ ] Add circuit breaker pattern for API calls
- [ ] Implement retry logic with exponential backoff

## Related Files

- Core Interfaces: `lib/core/interfaces/`
- Data Providers: `lib/services/providers/`
- Export Handlers: `lib/services/export/`
- Mock Implementations: `lib/services/mock/`

## License

This implementation follows SOLID principles and design patterns best practices.
